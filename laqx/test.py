import jax
import jax.numpy as jnp
import numpy as np
from laqx import networks, operators

from tqdm import tqdm
import os
import pickle
import sys

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from laqx.utils import checkpoint, distributed

jax.config.update("jax_debug_nans", True)

def train(args):
    if args.multi_host:
        num_hosts, host_idx = distributed.initialize_distributed_runtime()
    else:
        num_hosts, host_idx = 1, 0
    key = jax.random.PRNGKey(args.seed)
    num_devices = jax.local_device_count()
    # Device logging
    print('Starting LaQX Calculation with %i XLA devices per host '
                'across %i hosts.', num_devices, num_hosts)
    
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


    ex, ex2 = [], []
    def get_optimizer(args):
        local_energy = operators.operator_provider(network_apply, args, inference=True)
        batch_mcmc_step = operators.mcmc_provider(network_apply, args)

        def all_update(params, data, key):
            data, aux_mcmc = batch_mcmc_step(params, data, key)
            e_l, aux_local = local_energy(params, data, {'cache': aux_mcmc['cache']})
            logdict = {'pmove': jax.lax.pmean(jnp.mean(aux_mcmc['pmove']), axis_name='batch')}
            if args.fast_update:
                logdict['neighbor'] = jax.lax.pmax(aux_local['neighbor'], axis_name='batch')
            if args.reduce != 0:
                logdict['apply'] = jax.lax.pmax(aux_local['apply'], axis_name='batch')
            if args.reduce2 != 0:
                logdict['apply2'] = jax.lax.pmax(aux_local['apply2'], axis_name='batch')
            return data, e_l, logdict

        all_update = jax.jit(shard_map(all_update, mesh=mesh, in_specs=(None, pbatch, pbatch), out_specs=(pbatch, pbatch, pnone), check_rep=False))
        return all_update

    all_update = get_optimizer(args)
    optimizer_list = {}
    optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'] = all_update

    with tqdm(total=args.steps) as tq:
        for t in range(args.steps):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, args.batchsize)
            subkeys = jax.device_put(subkeys, jax.sharding.NamedSharding(mesh, pbatch))
            data, loss, logdict = optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'](params, data, subkeys)
        
            loss = jax.experimental.multihost_utils.process_allgather(loss, tiled=True)
            loss = loss.reshape(args.batchsize, -1)

            ex.append(np.mean(loss, axis=0))
            ex2.append(np.mean(np.abs(loss) ** 2, axis=0))

            if args.reduce != 0:
                args.reduce = (int(logdict['apply'] + 1.5 * args.pad)) // args.pad * args.pad
            
            if args.reduce2 != 0:
                args.reduce2 = (int(logdict['apply2'] + 1.5 * args.pad2)) // args.pad2 * args.pad2

            if args.fast_update:
                args.neighbor = int(logdict['neighbor']) + 1

            if f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}' not in optimizer_list:
                all_update = get_optimizer(args)
                optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'] = all_update
                print(f"New Reduce: {args.reduce}", file=sys.stderr)
                print(f"New Neighbor: {args.neighbor}", file=sys.stderr)
                print(f"New Reduce2: {args.reduce2}", file=sys.stderr)

            tq.set_postfix(logdict,refresh=False)
            tq.update(1)

    ex, ex2 = np.array(ex), np.array(ex2)
    mean = np.mean(ex, axis=0)
    x2_mean = np.mean(ex2, axis=0)
    variance = x2_mean - np.abs(mean) ** 2

    if args.obs in ["energy", "spin_up", "double", "polar"] and args.num_states == 1:
        if args.obs == 'energy':
            local_path = "result.txt"
        elif args.obs == 'polar':
            mean = np.angle(mean)
            local_path = "polar.txt"
        
        with open(f"{args.output}/{local_path}", 'a') as f:
            if args.symmetry == "T":
                print(f"kx: {args.kx} ky: {args.ky} Energy: {mean[0]} Variance: {variance[0]}", file=f)
            else:
                print(f"Energy: {mean[0]} Variance: {variance[0]}", file=f)

    else:
        if args.obs == "localo":
            local_path = "localo.npz"

        elif args.obs == "fermi":
            local_path = "fermi2.npz"

        elif args.obs == "energy":
            local_path = "energy.npz"

        with open(f"{args.output}/{local_path}", 'wb') as f:
            pickle.dump({'mean': mean, "variance": variance}, f)

    if host_idx == 0:
        print(f'Results saved to {args.output}/{local_path}')