import jax
import jax.numpy as jnp
from laqx import networks, operators
from laqx.utils import branch, checkpoint, distributed

import csv
from tqdm import tqdm
import os
import pickle

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import sys

jax.config.update("jax_debug_nans", True)

def train(args):
    if args.multi_host:
        num_hosts, host_idx = distributed.initialize_distributed_runtime()
    else:
        num_hosts, host_idx = 1, 0

    key = jax.random.PRNGKey(args.seed)

    num_devices = jax.local_device_count()
    total_devices = num_devices * num_hosts
    _, network_apply, _ = networks.network_provider(args)

    pbatch, pnone = P('batch'), P(None)
    mesh = jax.make_mesh((total_devices,), ('batch'))

    fname = checkpoint.find_last_checkpoint(args.output)

    with open(fname, 'rb') as f:
        ckpt_data = pickle.load(f)
        data, params = ckpt_data['data'], ckpt_data['params']
        print("Data shape loaded:", data.shape)
        data = data[:args.batchsize]
        if args.precision == "x64":
            params = jax.tree_map(lambda x: x.astype(jnp.float64), params)

    if args.use_x64:
        jax.config.update("jax_enable_x64", True)

    data = jnp.array(data)
    data = jax.device_put(data, jax.sharding.NamedSharding(mesh, pbatch))


    def get_mcmc_step(args):
        batch_mcmc_step = operators.mcmc_provider(network_apply, args)
        def pbatch_mcmc_step(params, data, key):
            data, aux = batch_mcmc_step(params, data, key)
            logdict = {'pmove': jax.lax.pmean(jnp.mean(aux['pmove']), axis_name='batch')}
            if args.fast_update:
                logdict['neighbor'] = jax.lax.pmax(jnp.max(aux['cache']['neighbor']), axis_name='batch')
            return data, logdict
        
        pbatch_mcmc_step = jax.jit(shard_map(pbatch_mcmc_step, mesh=mesh, in_specs=(None, pbatch, pbatch), out_specs=(pbatch, pnone), check_rep=False))
        return pbatch_mcmc_step

    pbatch_mcmc_step = get_mcmc_step(args)
    mcmc_list = {}
    mcmc_list[f'N{args.neighbor}'] = pbatch_mcmc_step
    
    with tqdm(total=args.drop_step) as tq:
        for t in range(args.drop_step):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, args.batchsize)
            subkeys = jax.device_put(subkeys, jax.sharding.NamedSharding(mesh, pbatch))
            data, logdict = mcmc_list[f'N{args.neighbor}'](params, data, subkeys)
            tq.set_postfix(logdict,refresh=False)
            if args.fast_update:
                args.neighbor = int(logdict['neighbor']) + 1

                if f'N{args.neighbor}' not in mcmc_list:
                    pbatch_mcmc_step = get_mcmc_step(args)
                    mcmc_list[f'N{args.neighbor}'] = pbatch_mcmc_step
                    print(f"New Neighbor: {args.neighbor}", file=sys.stderr)
            tq.update(1)


    local_energy = operators.operator_provider(network_apply, args)
    def batch_local_energy(params, data):
        e_l, aux_local = local_energy(params, data, {'cache': None})
        return jax.lax.pmean(jnp.mean(e_l), axis_name='batch')
    
    pbatch_local_energy = jax.jit(shard_map(batch_local_energy, mesh=mesh, in_specs=(None, pbatch), out_specs=pnone, check_rep=False))
    Et = pbatch_local_energy(params, data)
    weights = jnp.ones(args.batchsize)

    def get_gfmc_step(args):
        gfmc_update_fn = operators.gfmc_provider(network_apply, args)
        gfmc_update_fn = jax.jit(shard_map(gfmc_update_fn, mesh=mesh, in_specs=(None, pbatch, pbatch, None, pbatch), out_specs=(pbatch, pbatch, pnone, pnone), check_rep=False))
        return gfmc_update_fn

    pgfmc_update_fn = get_gfmc_step(args)
    gfmc_list = {}
    gfmc_list[f'N{args.neighbor}'] = pgfmc_update_fn

    local_path = f'{args.mode}.csv'
    with open(os.path.join(args.output, local_path), 'w') as csvfile:
        field = ['loss', 'offset', 'branch_change', 'mean_step']
        writer = csv.DictWriter(csvfile, fieldnames=field)
        writer.writeheader()
        with tqdm(total=args.steps) as tq:
            for t in range(args.steps):
                key, subkey, subkey2 = jax.random.split(key, 3)
                subkeys = jax.random.split(subkey, args.batchsize)
                subkeys = jax.device_put(subkeys, jax.sharding.NamedSharding(mesh, pbatch))
                if t != 0:
                    data = jax.device_put(data, jax.sharding.NamedSharding(mesh, pbatch))
                weights = jax.device_put(weights, jax.sharding.NamedSharding(mesh, pbatch))

                data, weights, Et, logdict = gfmc_list[f'N{args.neighbor}'](params, data, weights, Et, subkeys)

                data = jnp.array(jax.experimental.multihost_utils.process_allgather(data)).reshape(args.batchsize, -1)
                weights = jnp.array(jax.experimental.multihost_utils.process_allgather(weights)).reshape(args.batchsize)

                weights, data, num_to_change = branch.branch(weights, subkey2, data)
                logdict['branch_change'] = num_to_change

                if args.fast_update:
                    args.neighbor = int(logdict['neighbor']) + 1

                    if f'N{args.neighbor}' not in gfmc_list:
                        pgfmc_update_fn = get_gfmc_step(args)
                        gfmc_list[f'N{args.neighbor}'] = pgfmc_update_fn
                        print(f"New Neighbor: {args.neighbor}", file=sys.stderr)
                tq.set_postfix(logdict,refresh=False)
                logdict.pop('neighbor', None)
                logdict.pop('spin_up', None)
                writer.writerow(logdict)
                tq.update(1)
    
    
    data = jnp.array(jax.experimental.multihost_utils.process_allgather(data)).reshape(args.batchsize, -1)


    if host_idx == 0:
        local_csv_path = os.path.join(args.output, local_path)
        print(f'CSV saved to {local_csv_path}')