import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block

def init_nnb(key, args):
    hilbert_size = 2 * args.L1 * args.L2
    params, io = {}, {}
    key, subkey1, subkey2 = jax.random.split(key, 3)
    io['pe'] = jax.random.normal(subkey1, shape=(args.L1 * args.L2, args.hidden))
    io['embedding'] = jax.random.normal(subkey2, shape=(4, args.hidden))

    params['transformer'] = []
    for _ in range(args.layers):
        dic = {}
        key, subkey1, subkey2 = jax.random.split(key, 3)
        dic['attention'] = network_block.init_multihead_attention(subkey1, args.hidden)
        dic['MLP'] = network_block.init_linear_layer(subkey2, args.hidden, args.hidden)
        params['transformer'].append(dic)
    key, subkey  = jax.random.split(key)
    io['output'] = network_block.init_linear_layer(subkey, args.hidden, args.particles * args.ndet * 2)
    if args.dtype == 'complex':
        io['output_i'] = network_block.init_linear_layer(key, args.hidden, args.particles * args.ndet * 2)
    params['io'] = io
    return params

def apply_nnb(params, pos, cache, args):
    activation = network_block.get_activation(args.activation)
    ones = jnp.nonzero(pos, size=args.particles)
    pos = pos.reshape(2, -1).transpose(1, 0)

    indices = jnp.dot(pos, 2 ** jnp.arange(2))
    hidden = params['io']['pe'] + params['io']['embedding'][indices]

    for layer in params['transformer']:
        hidden = network_block.multihead_attention(hidden, args.num_head, **layer['attention']) + hidden
        hidden = activation(network_block.linear_layer(hidden, **layer['MLP'])) + hidden
    orbitals = network_block.linear_layer(hidden, **params['io']['output'])
    if args.dtype == 'complex':
        orbitals = orbitals + 1j * network_block.linear_layer(hidden, **params['io']['output_i'])
    orbitals = orbitals.reshape(args.L1 * args.L2, 2, args.ndet, -1).transpose(1, 0, 2, 3).reshape(2 * args.L1 * args.L2, args.ndet, -1)
    orbitals = orbitals[ones].transpose(1, 0, 2)
    sign, logdet = jnp.linalg.slogdet(orbitals)
    if sign.shape[0] == 1:
        if args.dtype == 'complex':
            logdet = jnp.log(sign) + logdet
            sign = jnp.ones_like(logdet, dtype=jnp.float32)
        cache = {}
        cache['sign'], cache['logdet'] = sign[0], logdet[0]
        return cache
    
    max_logdet = jax.lax.stop_gradient(jnp.max(logdet))
    det = jnp.exp(logdet - max_logdet)
    mask = jnp.abs(det) > 0.0
    result = jnp.sum(sign * det, where=mask)
    if args.dtype == 'complex':
        logdet = jnp.log(result) + max_logdet
        sign = jnp.ones_like(logdet, dtype=jnp.float32)
    else:
        sign = jnp.sign(result)
        logdet = jnp.log(jnp.abs(result)) + max_logdet
    cache = {}
    cache['sign'], cache['logdet'] = sign, logdet
    return cache

def get_orbitals(params, pos, args):
    activation = network_block.get_activation(args.activation)
    # ones = jnp.nonzero(pos, size=args.particles)
    pos = pos.reshape(2, -1).transpose(1, 0)
    indices = jnp.dot(pos, 2 ** jnp.arange(2))
    hidden = params['io']['pe'] + params['io']['embedding'][indices]

    for layer in params['transformer']:
        hidden = network_block.multihead_attention(hidden, args.num_head, **layer['attention']) + hidden
        hidden = activation(network_block.linear_layer(hidden, **layer['MLP'])) + hidden
    orbitals = network_block.linear_layer(hidden, **params['io']['output'])
    if args.dtype == 'complex':
        orbitals = orbitals + 1j * network_block.linear_layer(hidden, **params['io']['output_i'])
    orbitals = orbitals.reshape(args.L1 * args.L2, 2, args.ndet, -1).transpose(1, 0, 2, 3).reshape(2 * args.L1 * args.L2, args.ndet, -1)
    orbitals = orbitals.transpose(1, 0, 2)
    return orbitals

def make_transformer(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, args=args)
    def symm_apply(params, data, cache):
        if not args.symmetry:
            return apply(params, data, cache)
        p_apply = partial(apply, cache=None)

        data = data.reshape(2, args.L1, args.L2)

        data, weight = network_block.get_symmetry([data], [1], args)

        batched_data = jnp.stack(data, axis=0).reshape(-1, 2 * args.L1 * args.L2)
        weight = jnp.array(weight)
        print("symmetry operation:", batched_data.shape[0])
        
        new_cache = jax.vmap(p_apply, in_axes=(None, 0))(params, batched_data)
        out_cache = {}

        max_logdet = jax.lax.stop_gradient(jnp.max(new_cache['logdet']))
        det = jnp.exp(new_cache['logdet'] - max_logdet)
        mask = jnp.abs(det) > 0.0
        result = jnp.sum(new_cache['sign'] * det * weight, where=mask)
        if args.dtype == 'complex':
            logdet = jnp.log(result) + max_logdet
            sign = jnp.ones_like(logdet, dtype=jnp.float32)
        else:
            sign = jnp.sign(result)
            logdet = jnp.log(jnp.abs(result)) + max_logdet
        out_cache['sign'], out_cache['logdet'] = sign, logdet
        return out_cache
    orbitals = partial(get_orbitals, args=args)
    return init, symm_apply, orbitals