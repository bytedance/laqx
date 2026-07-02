import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block

def init_nnb(key, args):
    params = {}
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2

    key, subkey = jax.random.split(key)
    params['embedding'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, args.hidden))

    params['cnn'] = []
    for _ in range(args.layers):
        dic = {}
        key, subkey = jax.random.split(key)
        dic['norm'] = network_block.init_layernorm(args.hidden)
        dic['cnn'] = network_block.init_convolution(subkey, args.hidden, args.hidden, args.cutoff)
        params['cnn'].append(dic)

    key, subkey = jax.random.split(key)
    params['MLP'] = network_block.init_MLP(subkey, args.hidden, args.MLP_hidden, args.MLP_layers)

    output_dim = physical * args.mpsdim ** 2 * args.mps_num_head
    key, subkey = jax.random.split(key)
    params['output'] = network_block.init_mps(subkey, args.MLP_hidden, output_dim)
    return params


def apply_nnb(params, pos, cache, args):
    cache = {}
    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2

    indices = network_block.position_to_embedding_indices(pos, args, physical)
    hidden = params['embedding'][indices]

    hidden = hidden.reshape(args.L1, args.L2, args.hidden)
    for i in range(args.layers):
        hidden = hidden + activation(network_block.convolution(hidden, args.boundary1, args.boundary2, **params['cnn'][i]['cnn'], cutoff=args.cutoff))
        if i == args.layers - 1:
            hidden = hidden.astype(jnp.float64)
        hidden = network_block.layernorm(hidden, **params['cnn'][i]['norm'])
    hidden = hidden.reshape(args.L1 * args.L2, args.hidden)

    if args.precision != 'x64':
        hidden = hidden.astype(jnp.float32)
    hidden = network_block.apply_MLP(hidden, params['MLP'], activation)

    mps = network_block.linear_layer(hidden, **params['output'])
    mps = mps.reshape(args.L1 * args.L2, physical, args.mpsdim, args.mpsdim, args.mps_num_head)

    x = pos.reshape(2, args.L1 * args.L2)[0]
    mps_x = mps[jnp.arange(x.shape[0]), x]
    sign, logdet = network_block.multi_mps_contraction_pbc(mps_x)
    cache['sign'], cache['logdet'] = sign, logdet
    return cache


def make_cnn_mps(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, cache=None, args=args)

    def symm_apply(params, data, cache):
        if not args.symmetry:
            return apply(params, data)

        data = data.reshape(2, args.L1, args.L2)
        data, weight = network_block.get_symmetry([data], [1], args)
        batched_data = jnp.stack(data, axis=0).reshape(-1, 2 * args.L1 * args.L2)

        weight = jnp.array(weight)
        if args.use_boson:
            weight = jnp.ones_like(weight)

        new_cache = jax.vmap(apply, in_axes=(None, 0))(params, batched_data)
        sign, logdet = network_block.combine_signed_logdet(new_cache['sign'], new_cache['logdet'], args, weight=weight)
        return {'sign': sign, 'logdet': logdet}

    return init, symm_apply, None
