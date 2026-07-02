import jax
import jax.numpy as jnp
from laqx import networks, operators

import optax
from tqdm import tqdm
import os
import pickle
import copy

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map


jax.config.update("jax_debug_nans", True)

def train(args, params, key, num_hosts, host_idx):
    num_devices = jax.local_device_count()
    total_devices = num_devices * num_hosts

    _, _, orbitals = networks.network_provider(args)
    new_args = copy.deepcopy(args)
    new_args.network_name = args.restore_network_name
    new_args.ndet, new_args.layers, new_args.hidden, new_args.num_head = args.restore_ndet, args.restore_layers, args.restore_hidden, args.restore_head
    _, target_apply, target_orbitals = networks.network_provider(new_args)
    orbitals, target_orbitals = jax.vmap(orbitals, in_axes=(None, 0)), jax.vmap(target_orbitals, in_axes=(None, 0))

    with open(args.restore, 'rb') as f:
        ckpt_data = pickle.load(f)
        target_params = ckpt_data['params']
        if args.precision == "x64":
            target_params = jax.tree_map(lambda x: x.astype(jnp.float64), target_params)
        data = ckpt_data['data']
        print("pretrain", data.shape)

    batch_mcmc_step = operators.mcmc_provider(target_apply, args)

    def learning_rate_schedule(t_):
        return args.lr * (1.0 / (1.0 + (t_ / 10000)))
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(learning_rate_schedule), optax.scale(-1))
    opt_state = optimizer.init(params)

    pbatch, pnone = P('batch'), P(None)
    mesh = jax.make_mesh((total_devices,), ('batch'))

    data = jnp.array(data)
    data = jax.device_put(data, jax.sharding.NamedSharding(mesh, pbatch))

    if args.num_states > 1:
        orbitals, target_orbitals = jax.vmap(orbitals, in_axes=(None, 0)), jax.vmap(target_orbitals, in_axes=(None, 0))


    def get_ratio(data):
        norm = jax.lax.pmean(jnp.mean(jnp.sqrt(jnp.sum(jnp.abs(orbitals(params, data)) ** 2, axis=(2,3)))), axis_name='batch')
        target_norm = jax.lax.pmean(jnp.mean(jnp.sqrt(jnp.sum(jnp.abs(target_orbitals(target_params, data)) ** 2, axis=(2,3)))), axis_name='batch')
        ratio = norm / target_norm
        return ratio
    get_ratio = jax.jit(shard_map(get_ratio, mesh=mesh, in_specs=(pbatch,), out_specs=pnone, check_rep=False))
    ratio = get_ratio(data)
    print(f"Norm_Ratio:{ratio}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    def loss_fn(pp, dd):
        prediction, answers = orbitals(pp, dd), target_orbitals(target_params, dd)
        loss = jax.lax.pmean(jnp.mean(jnp.abs(prediction - answers * ratio) ** 2), axis_name='batch')
        return loss
    val_and_grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    def all_update(subkeys, params, data, opt_state):
        data, aux = batch_mcmc_step(target_params, data, subkeys)
        loss, grad = val_and_grad_fn(params, data)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        logdict = {'loss': loss, 'pmove': jax.lax.pmean(jnp.mean(aux['pmove']), axis_name='batch')}
        return params, opt_state, data, logdict
    all_update = jax.jit(shard_map(all_update, mesh=mesh, in_specs=(pbatch, None, pbatch, None), out_specs=(pnone, pnone, pbatch, pnone), check_rep=False))

    with tqdm(total=args.steps) as tq:
        for t in range(args.steps):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, args.batchsize)
            subkeys = jax.device_put(subkeys, jax.sharding.NamedSharding(mesh, pbatch))
            params, opt_state, data, logdict = all_update(subkeys, params, data, opt_state)

            tq.set_postfix(logdict,refresh=False)
            tq.update(1)

    local_path = os.path.join(args.output, "ckpt_000000.npz")
    with open(local_path, 'wb') as f:
        all_data = jax.experimental.multihost_utils.process_allgather(data)
        if args.num_states == 1:
            all_data = all_data.reshape((-1,data.shape[-1]))
        else:
            all_data = all_data.reshape((-1, args.num_states, data.shape[-1]))
        pickle.dump({'t': -1, 'data': all_data, 'params': params}, f)

    if host_idx == 0:
        print(f'Pretrain checkpoint saved: {local_path}')
