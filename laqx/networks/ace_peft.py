import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block

def init_nnb(key, args, params=None):
    params_peft = {'output': params['output']}
    if 'output_i' in params:
        params_peft['output']['w'] += 1j * params['output_i']['w']
        params_peft['output']['b'] += 1j * params['output_i']['b']

    params.pop('output', None)
    params.pop('output_i', None)
    return {'fixed': params, 'peft': jax.tree_map(lambda x: x.astype(jnp.complex64), params_peft)}

def apply_nnb(params, pos, cache, args):
    cache = {}
    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    ones = network_block.get_occupied_indices(pos, args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    hidden = params['fixed']['embedding'][indices]

    hidden = hidden.reshape(args.L1, args.L2, args.hidden)
    for i in range(args.layers):
        hidden = hidden + activation(network_block.convolution(hidden, args.boundary1, args.boundary2, **params['fixed']['cnn'][i]['cnn'], cutoff=args.cutoff))
        if i == args.layers - 1:
            hidden = hidden.astype(jnp.float64)
        hidden = network_block.layernorm(hidden, **params['fixed']['cnn'][i]['norm'])
    hidden = hidden.reshape(args.L1 * args.L2, args.hidden)
    if args.precision != 'x64':
        hidden = hidden.astype(jnp.float32)
    hidden = network_block.apply_MLP(hidden, params['fixed']['MLP'], activation)

    orbitals = network_block.linear_layer(hidden, **params['peft']['output'])

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, args.ndet, -1).transpose(1, 0, 2, 3).reshape(spin_channels * args.L1 * args.L2, args.ndet, -1)
    orbitals = orbitals[ones].transpose(1, 0, 2)

    sign, logdet = jnp.linalg.slogdet(orbitals)

    if sign.shape[0] == 1:
        if args.dtype == 'complex':
            logdet = jnp.log(sign) + logdet
            sign = jnp.ones_like(logdet, dtype=jnp.float32)
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
    cache['sign'], cache['logdet'] = sign, logdet
    return cache

def make_ace_peft(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, cache=None, args=args)
    def symm_apply(params, data, cache):
        return network_block.symmetrize_apply(params, data, cache, args, apply)
    return init, symm_apply, None