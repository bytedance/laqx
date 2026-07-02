import jax
import jax.numpy as jnp
import numpy as np
from laqx import networks, operators

import csv
from tqdm import tqdm
import os
import pickle
import copy
import sys
from laqx.pretrain import train as pretrain_train

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from laqx.utils import adam, checkpoint, distributed, march, spring

jax.config.update("jax_debug_nans", True)

def init_electrons(key, args):
    base_array_up = jnp.array([1] * args.particles_up + [0] * (args.L1 * args.L2 - args.particles_up))
    base_array_down = jnp.array([1] * (args.particles - args.particles_up) + [0] * (args.L1 * args.L2 - (args.particles - args.particles_up)))
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(key, args.batchsize * args.num_states)
    subkeys = jax.random.split(subkey, args.batchsize * args.num_states)
    batchup = jax.vmap(lambda k: jax.random.permutation(k, base_array_up))(keys)
    batchdown = jax.vmap(lambda k: jax.random.permutation(k, base_array_down))(subkeys)
    output = jnp.concatenate([batchup, batchdown], axis=1).astype(jnp.int8)
    if args.num_states > 1:
        output = output.reshape(args.batchsize, args.num_states, -1)
    return output


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
    
    network_init, network_apply, _ = networks.network_provider(args)
    key, subkey = jax.random.split(key)
    params = network_init(subkey)
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print("Parameter Size:", param_count)

    # for the data initialization, we want to use different key 
    # for different hosts.
    key, subkey = jax.random.split(key)
    data = init_electrons(subkey, args)


    pbatch, pnone = P('batch'), P(None)
    mesh = jax.make_mesh((total_devices,), ('batch'))

    def get_optimizer(args):
        local_energy = operators.operator_provider(network_apply, args)
        batch_mcmc_step = operators.mcmc_provider(network_apply, args)
            
        if args.mode == "spring":
            update_fn = spring.get_spring_update_fn(network_apply, local_energy, args, batch_mcmc_step, num_hosts * num_devices)
            opt_state = jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.zeros_like(x), params))[0]
        elif args.mode == "adam":
            update_fn, optimizer = adam.get_adam_update_fn(network_apply, local_energy, args, batch_mcmc_step, num_hosts * num_devices)
            opt_state = optimizer.init(params)
        elif args.mode == "march":
            update_fn = march.get_march_update_fn(network_apply, local_energy, args, batch_mcmc_step, num_hosts * num_devices)
            opt_state = (jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.zeros_like(x), params))[0], jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.ones_like(x) * args.v_init, params))[0])

        update_fn = jax.jit(shard_map(update_fn, mesh=mesh, in_specs=(None, None, pbatch, None, pbatch), out_specs=(pnone, pnone, pbatch, pnone), check_rep=False))
        return update_fn, opt_state

    t_init = 0
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    fname = checkpoint.find_last_checkpoint(args.output)
    if fname is None and args.restore:
        pretrain_fname = checkpoint.find_last_checkpoint(args.restore)
        pretrain_args = copy.deepcopy(args)
        pretrain_args.steps, pretrain_args.mcmc_step, pretrain_args.lr, pretrain_args.restore = args.pretrain_step, 30, 3e-4, pretrain_fname
        pretrain_args.mode = "pretrain"   
        key, subkey = jax.random.split(key)
        pretrain_train(pretrain_args, params, subkey, num_hosts, host_idx)
        fname = checkpoint.find_last_checkpoint(args.output)

    update_fn, opt_state = get_optimizer(args)
        
    if fname:
        with open(fname, 'rb') as f:
            ckpt_data = pickle.load(f)
            if ckpt_data['t'] < 0:
                print("loading from a pretrain checkpoint")
                params, data = ckpt_data['params'], ckpt_data['data']
            else:
                print("loading from a vmc checkpoint")
                t_init, data, params, opt_state_old = ckpt_data['t'] + 1, ckpt_data['data'], ckpt_data['params'], ckpt_data['opt_state']
                if not args.reset_opt:
                    opt_state = opt_state_old
                with open(os.path.join(args.output, "log.csv"), 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    logging = [row for row in reader][:t_init]
            data = data[:args.batchsize]
            if args.precision == "x64":
                params = jax.tree_map(lambda x: x.astype(jnp.float64), params)

    if args.use_x64:
        jax.config.update("jax_enable_x64", True)

    data = jnp.array(data)
    data = jax.device_put(data, jax.sharding.NamedSharding(mesh, pbatch))
    print("After Sharding", data.shape)

    if args.burn_in:   
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


    optimizer_list = {}
    optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'] = update_fn

    if args.debug:
        from ctypes import cdll
        libcudart = cdll.LoadLibrary('libcudart.so')

    with open(os.path.join(args.output, "log.csv"), 'w') as csvfile:
        f_name = ['loss', 'pmove', 'variance', 'lr']
        if args.mode != 'adam':
            f_name.append('norm')
        writer = csv.DictWriter(csvfile, fieldnames=f_name)
        writer.writeheader()
        if t_init > 0:
            for row in logging:
                writer.writerow(row)
        with tqdm(total=args.steps, initial=t_init) as tq:
            for t in range(t_init, args.steps):
                if args.debug and t == 3:
                    libcudart.cudaProfilerStart()
                key, subkey = jax.random.split(key)
                subkeys = jax.random.split(subkey, args.batchsize)
                subkeys = jax.device_put(subkeys, jax.sharding.NamedSharding(mesh, pbatch))
                params, opt_state, data, logdict = optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'](params, opt_state, data, t, subkeys)
                
                if args.reduce != 0:
                    args.reduce = (int(logdict['apply'] + 1.5 * args.pad)) // args.pad * args.pad
                
                if args.reduce2 != 0:
                    args.reduce2 = (int(logdict['apply2'] + 1.5 * args.pad2)) // args.pad2 * args.pad2
                        
                if args.fast_update:
                    args.neighbor = int(logdict['neighbor']) + 1
                
                if f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}' not in optimizer_list:
                    update_fn, _ = get_optimizer(args)
                    optimizer_list[f'R{args.reduce}_N{args.neighbor}_D{args.reduce2}'] = update_fn
                    print(f"New Reduce: {args.reduce}", file=sys.stderr)
                    print(f"New Neighbor: {args.neighbor}", file=sys.stderr)
                    print(f"New Reduce2: {args.reduce2}", file=sys.stderr)

            
                if not args.fast_update:
                    logdict.pop('neighbor', None)

                tq.set_postfix(logdict,refresh=False)
                logdict.pop('apply', None)
                logdict.pop('apply2', None)
                logdict.pop('neighbor', None)
                logdict.pop('spin_up', None)
                logdict.pop('spin_other', None)
                writer.writerow(logdict)
                tq.update(1)

                if ((t + 1) % args.save_frequency == 0):
                    local_path = os.path.join(args.output, f"ckpt_{t+1:06d}.npz")
                    all_data = jax.experimental.multihost_utils.process_allgather(data)
                    if args.num_states == 1:
                        all_data = all_data.reshape((-1,data.shape[-1]))
                    else:
                        all_data = all_data.reshape((-1, args.num_states, data.shape[-1]))

                    with open(local_path, 'wb') as f:
                        pickle.dump({'t': t, 'data': all_data, 'params': params, 'opt_state': opt_state}, f)
                        
                    csvfile.flush()

                    if host_idx == 0:
                        print(f'Checkpoint saved: {local_path}')

    if args.debug:
        libcudart.cudaProfilerStop()

