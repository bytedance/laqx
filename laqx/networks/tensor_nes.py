import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block
  
def init_nnb(key, args):
    params = {}
    key, subkey = jax.random.split(key, 2)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    params['embedding'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, spin_channels * args.particles * args.num_states))
    if args.dtype == 'complex':
        key, subkey = jax.random.split(key, 2)
        params['embedding_i'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, spin_channels * args.particles * args.num_states))
    return params

def apply_nnb(params, pos, args):
    cache = {}
    spin_channels = network_block.get_spin_channels(args)
    ones = network_block.get_occupied_indices(pos, args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    orbitals = params['embedding'][indices]
    if args.dtype == 'complex':
        orbitals = orbitals + 1j * params['embedding_i'][indices]

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    orbitals = orbitals[ones]

    orbitals = orbitals.reshape(args.particles, args.num_states, -1).transpose(1, 0, 2)
    sign, logdet = jnp.linalg.slogdet(orbitals)

    logdet = logdet.astype(jnp.float64) if args.dtype != 'complex' else logdet.astype(jnp.complex128)
    cache['psi'] = sign * jnp.exp(logdet)
    return cache

def get_orbitals(params, pos, args):
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    orbitals = params['embedding'][indices]
    if args.dtype == 'complex':
        orbitals = orbitals + 1j * params['embedding_i'][indices]

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    return orbitals


def make_tensor_nes(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, args=args)
    def symm_apply(params, data, cache):
        return network_block.symmetrize_apply(params, data, cache, args, apply)
    orbitals = partial(get_orbitals, args=args)
    return init, symm_apply, orbitals
