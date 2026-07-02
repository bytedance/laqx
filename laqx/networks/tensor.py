import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block
  
def init_nnb(key, args):
    params = {}
    key, subkey = jax.random.split(key, 2)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    params['embedding'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, spin_channels * args.particles // args.ndet))
    if args.dtype == 'complex':
        key, subkey = jax.random.split(key, 2)
        params['embedding_i'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, spin_channels * args.particles // args.ndet))
    return params

def apply_nnb(params, pos, args):
    cache = {}
    spin_channels = network_block.get_spin_channels(args)
    ones = network_block.get_occupied_indices(pos, args)
    Nu = jnp.sum(pos.reshape(2, -1)[0])
    cache['pos'] = pos
    cache['ones'] = ones
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    orbitals = params['embedding'][indices]
    if args.dtype == 'complex':
        orbitals = orbitals + 1j * params['embedding_i'][indices]

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    orbitals = orbitals[ones]

    if args.ndet == 1:
        orbitals = orbitals[None]
    elif args.ndet == 2:
        orbitals = orbitals.reshape(2, args.particles // args.ndet, args.particles // args.ndet)

    if args.fast_update:
        orbitals = orbitals.astype(jnp.float64) if args.dtype != 'complex' else orbitals.astype(jnp.complex128)
        inv, (sign, logdet) = jax.vmap(network_block.compute_inv_det_lu)(orbitals)
        cache['det'] = (inv, sign, logdet)
    else:
        sign, logdet = jnp.linalg.slogdet(orbitals)

    sign = jnp.prod(sign)
    logdet = jnp.sum(logdet)
    if args.dtype == 'complex':
        logdet = jnp.log(sign) + logdet
        sign = jnp.ones_like(logdet, dtype=jnp.float32)
    cache['sign'], cache['logdet'] = sign, logdet
    cache['neighbor'] = jnp.zeros((), dtype=sign.dtype)
    return cache

def apply_nnb_fast(params, pos, cache, args):
    old_pos = cache['pos']
    spin_channels = network_block.get_spin_channels(args)
    Nu = jnp.sum(pos.reshape(2, -1)[0])

    creation = jnp.nonzero(pos - old_pos == 1, size=2, fill_value=-1)[0]
    annihilation = jnp.nonzero(pos - old_pos == -1, size=2, fill_value=-1)[0]
    new_ones = jnp.where(cache['ones'] == annihilation[0], creation[0], cache['ones'])
    new_ones = jnp.where(new_ones == annihilation[1], creation[1], new_ones)
    
    new_cache = {}
    new_cache['pos'] = pos
    new_cache['ones'] = new_ones

    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    old_indices = network_block.position_to_embedding_indices(old_pos, args, physical)

    if args.dtype == 'complex':
        orbitals_new = params['embedding'][indices] + 1j * params['embedding_i'][indices]
        orbitals_old = params['embedding'][old_indices] + 1j * params['embedding_i'][old_indices]
    else:
        orbitals_new = params['embedding'][indices]
        orbitals_old = params['embedding'][old_indices]

    matrix_old = orbitals_old.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    matrix_old = matrix_old[cache['ones']]
    matrix_new = orbitals_new.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    matrix_new = matrix_new[new_ones]

    matrix_old = matrix_old.astype(jnp.float64) if args.dtype != 'complex' else matrix_old.astype(jnp.complex128)
    matrix_new = matrix_new.astype(jnp.float64) if args.dtype != 'complex' else matrix_new.astype(jnp.complex128)

    if args.ndet == 1:
        matrix_old = matrix_old[None]
        matrix_new = matrix_new[None]
        max_change = 3
    elif args.ndet == 2:
        matrix_old = matrix_old.reshape(2, args.particles // args.ndet, args.particles // args.ndet)
        matrix_new = matrix_new.reshape(2, args.particles // args.ndet, args.particles // args.ndet)
        max_change = 2

    def all_update(M_old, M_new, cache_det):
        changed_indices, neighbor = network_block.find_changed_columns(M_old[None], M_new[None], max_change)
        inv, sign, logdet = network_block.fast_update(M_old, M_new, cache_det, changed_indices)
        # sign, logdet = jnp.linalg.slogdet(M_new)
        return (inv, sign, logdet), neighbor

    (inv, sign, logdet), neighbor = jax.vmap(all_update)(matrix_old, matrix_new, cache['det'])
    neighbor = jnp.max(neighbor)
    new_cache['det'] = (inv, sign, logdet)

    cache['neighbor'] = jnp.maximum(cache['neighbor'], neighbor)
    new_cache['neighbor'] = cache['neighbor']

    sign = jnp.prod(sign)
    logdet = jnp.sum(logdet)
    if args.dtype == 'complex':
        logdet = jnp.log(sign) + logdet
        sign = jnp.ones_like(logdet, dtype=jnp.float32)

    new_cache['sign'], new_cache['logdet'] = sign, logdet
    return new_cache

def get_orbitals(params, pos, args):
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    orbitals = params['embedding'][indices]

    if args.dtype == 'complex':
        orbitals = orbitals + 1j * params['embedding_i'][indices]

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, -1).transpose(1, 0, 2).reshape(spin_channels * args.L1 * args.L2, -1)
    if args.ndet == 2:
        orbitals = jnp.split(orbitals, args.ndet)
        orbitals = jax.scipy.linalg.block_diag(*orbitals)
    return orbitals[None]


def make_tensor(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, args=args)
    orbitals = partial(get_orbitals, args=args)
    apply_fast = partial(apply_nnb_fast, args=args)

    def symm_apply(params, data, cache):
        return network_block.symmetrize_apply(params, data, cache, args, apply, apply_fast)

    return init, symm_apply, orbitals
