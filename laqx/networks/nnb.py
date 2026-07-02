import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block

def init_nnb(key, args):
    hilbert_size = 2 * args.L1 * args.L2
    params = {}
    key, subkey = jax.random.split(key)
    params['input'] = network_block.init_linear_layer(subkey, hilbert_size, args.hidden)
    params['MLP'] = []
    for _ in range(args.layers):
        key, subkey = jax.random.split(key)
        params['MLP'].append(network_block.init_linear_layer(subkey, args.hidden, args.hidden))
    key, subkey = jax.random.split(key, 2)
    params['output'] = network_block.init_linear_layer(key, args.hidden, hilbert_size * args.particles // args.ndet)
    if args.dtype == 'complex':
        params['output_i'] = network_block.init_linear_layer(subkey, args.hidden, hilbert_size * args.particles // args.ndet)
    return params

def apply_nnb(params, pos, cache, args):
    hidden = jax.nn.silu(network_block.linear_layer(pos, **params['input']))
    for layer in params['MLP']:
        hidden = jax.nn.silu(network_block.linear_layer(hidden, **layer))
    M = network_block.linear_layer(hidden, **params['output'])
    if args.dtype == 'complex':
        M = M + 1j * network_block.linear_layer(hidden, **params['output_i'])
    M = M.reshape((2 * args.L1 * args.L2, args.particles // args.ndet))
    
    ones = jnp.nonzero(pos, size=args.particles)
    matrix = M[ones]
    if args.ndet == 1:
        sign, logdet = jnp.linalg.slogdet(matrix)
    elif args.ndet == 2:
        matrix = matrix.reshape(2, args.particles // args.ndet, args.particles // args.ndet)
        matrix = jnp.matmul(matrix[0], matrix[1])
        sign, logdet = jnp.linalg.slogdet(matrix)
    if args.dtype == 'complex':
        logdet = jnp.log(sign) + logdet
        sign = jnp.ones_like(logdet, dtype=jnp.float32)
    cache = {}
    cache['sign'], cache['logdet'] = sign, logdet
    return cache

def get_orbitals(params, pos, args):
    hidden = jax.nn.silu(network_block.linear_layer(pos, **params['input']))
    for layer in params['MLP']:
        hidden = jax.nn.silu(network_block.linear_layer(hidden, **layer))
    M = network_block.linear_layer(hidden, **params['output'])
    if args.dtype == 'complex':
        M = M + 1j * network_block.linear_layer(hidden, **params['output_i'])
    M = M.reshape((2 * args.L1 * args.L2, args.particles // args.ndet))

    if args.ndet == 1:
        matrix = M
    elif args.ndet == 2:
        matrix = jnp.split(M, args.ndet)
        matrix = jax.scipy.linalg.block_diag(*matrix)
    return matrix[None]

def make_nnb(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, args=args)
    orbitals = partial(get_orbitals, args=args)
    return init, apply, orbitals