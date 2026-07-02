import jax
import jax.numpy as jnp
from functools import partial
from laqx.networks import network_block

def init_nnb(key, args):
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    params = {}
    key, subkey = jax.random.split(key)
    params['embedding'] = jax.random.normal(subkey, shape=(physical * args.L1 * args.L2, args.hidden))

    key, subkey = jax.random.split(key)
    params['cnn'] = network_block.init_convolution(subkey, args.hidden, args.hidden, args.cutoff)

    key, subkey = jax.random.split(key)
    params['MLP'] = network_block.init_MLP(subkey, args.hidden, args.MLP_hidden, args.MLP_layers)

    key, subkey = jax.random.split(key)
    params['output'] = network_block.init_linear_layer(subkey, args.MLP_hidden, args.particles * args.ndet * spin_channels)
    return params

def apply_nnb(params, pos, args):
    cache = {}
    cache['pos'] = pos
    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    ones = network_block.get_occupied_indices(pos, args)
    cache['ones'] = ones
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    hidden = params['embedding'][indices]

    hidden = hidden.reshape(args.L1, args.L2, args.hidden)
    hidden = hidden + activation(network_block.convolution(hidden, args.boundary1, args.boundary2, **params['cnn'], cutoff=args.cutoff))
    hidden = hidden.reshape(args.L1 * args.L2, args.hidden)
    
    hidden = network_block.apply_MLP(hidden, params['MLP'], activation)

    orbitals = network_block.linear_layer(hidden, **params['output'])
    cache['orbitals'] = orbitals

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, args.ndet, -1).transpose(1, 0, 2, 3).reshape(spin_channels * args.L1 * args.L2, args.ndet, -1)
    orbitals = orbitals[ones].transpose(1, 0, 2)

    if args.fast_update:
        inv, (sign, logdet) = jax.vmap(network_block.compute_inv_det_lu)(orbitals.astype(jnp.float64))
        cache['det'] = (inv, sign, logdet)
    else:
        sign, logdet = jnp.linalg.slogdet(orbitals)

    if sign.shape[0] == 1:
        sign, logdet = sign[0], logdet[0]
        cache['sign'], cache['logdet'] = sign, logdet
        cache['neighbor'] = jnp.zeros_like(sign)
        return cache

    max_logdet = jax.lax.stop_gradient(jnp.max(logdet))
    det = jnp.exp(logdet - max_logdet)
    mask = jnp.abs(det) > 0.0
    result = jnp.sum(sign * det, where=mask)
    sign = jnp.sign(result)
    logdet = jnp.log(jnp.abs(result)) + max_logdet
    cache['sign'], cache['logdet'] = sign, logdet

    cache['neighbor'] = jnp.zeros_like(sign)
    return cache

def apply_nnb_fast(params, pos, cache, args):
    old_pos = cache['pos']
    exchange = jnp.nonzero(pos - old_pos, size=2)[0]

    creation = jnp.nonzero(pos - old_pos == 1, size=2, fill_value=-1)[0]
    annihilation = jnp.nonzero(pos - old_pos == -1, size=2, fill_value=-1)[0]
    new_ones = jnp.where(cache['ones'] == annihilation[0], creation[0], cache['ones'])
    new_ones = jnp.where(new_ones == annihilation[1], creation[1], new_ones)

    new_cache = {}
    new_cache['pos'] = pos

    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)

    new_cache['ones'] = new_ones

    hidden_o = params['embedding'][indices]
    hidden = hidden_o.reshape(args.L1, args.L2, args.hidden)
    site = exchange % (args.L1 * args.L2)

    if args.dmax == 0:
        affect_site, is_valid = network_block.get_nearby(site, args) # 
        
        conv = lambda s: network_block.nearby_convolution(s, hidden, args, **params['cnn'])
        conv = jax.vmap(conv)(site).reshape(2, 9, args.hidden) # (2, 9, H)
        conv = jnp.where(is_valid.reshape(2, 9, 1), conv, conv[:, 4:5]).reshape(18, args.hidden)
    else:
        affect_site, is_valid = network_block.get_nearby_tprime(site, args) # 
        
        conv = network_block.nearby_convolution_tprime(site, hidden, args, **params['cnn']).reshape((args.cutoff + 1) ** 2, args.hidden)
        # conv = jax.vmap(conv)(site).reshape(2, 9, args.hidden) # (2, 9, H)
        if args.cutoff == 3:
            conv = jnp.where(is_valid.reshape(16, 1), conv, conv[5:6])
        elif args.cutoff == 5:
            conv = jnp.where(is_valid.reshape(36, 1), conv, conv[14:15])

    hidden = hidden_o[affect_site] + activation(conv)

    hidden = network_block.apply_MLP(hidden, params['MLP'], activation)
    
    orbitals = network_block.linear_layer(hidden, **params['output'])
    orbitals_old = cache["orbitals"]
    orbitals_new = orbitals_old.at[affect_site].set(orbitals)
    new_cache["orbitals"] = orbitals_new


    matrix_old = orbitals_old.reshape(-1, spin_channels, args.ndet, args.particles).transpose(1, 0, 2, 3).reshape(-1, args.ndet, args.particles)
    matrix_old = matrix_old[cache['ones']].transpose(1, 0, 2).astype(jnp.float64)
    matrix_new = orbitals_new.reshape(-1, spin_channels, args.ndet, args.particles).transpose(1, 0, 2, 3).reshape(-1, args.ndet, args.particles)
    matrix_new = matrix_new[new_ones].transpose(1, 0, 2).astype(jnp.float64)

    changed_indices, neighbor = network_block.find_changed_columns(matrix_old, matrix_new, args.neighbor)
    inv, sign, logdet = jax.vmap(network_block.fast_update, in_axes=(0, 0, 0, None))(matrix_old, matrix_new, cache['det'], changed_indices)
    new_cache['det'] = (inv, sign, logdet)

    cache['neighbor'] = jnp.maximum(cache['neighbor'], neighbor)
    new_cache['neighbor'] = cache['neighbor']

    if sign.shape[0] == 1:
        sign, logdet = sign[0], logdet[0]
        new_cache['sign'], new_cache['logdet'] = sign, logdet
        return new_cache

    max_logdet = jax.lax.stop_gradient(jnp.max(logdet))
    det = jnp.exp(logdet - max_logdet)
    mask = jnp.abs(det) > 0.0
    result = jnp.sum(sign * det, where=mask)
    sign = jnp.sign(result)
    logdet = jnp.log(jnp.abs(result)) + max_logdet
    new_cache['sign'], new_cache['logdet'] = sign, logdet
    return new_cache

def get_orbitals(params, pos, args):
    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    indices = network_block.position_to_embedding_indices(pos, args, physical)
    hidden = params['embedding'][indices]

    hidden = hidden.reshape(args.L1, args.L2, args.hidden)
    hidden = hidden + activation(network_block.convolution(hidden, args.boundary1, args.boundary2, **params['cnn'], cutoff=args.cutoff))
    hidden = hidden.reshape(args.L1 * args.L2, args.hidden)
    
    hidden = network_block.apply_MLP(hidden, params['MLP'], activation)

    orbitals = network_block.linear_layer(hidden, **params['output'])

    orbitals = orbitals.reshape(args.L1 * args.L2, spin_channels, args.ndet, -1).transpose(1, 0, 2, 3).reshape(spin_channels * args.L1 * args.L2, args.ndet, -1)
    orbitals = orbitals.transpose(1, 0, 2)

    return orbitals

def make_scale(args):
    init = partial(init_nnb, args=args)
    apply = partial(apply_nnb, args=args)
    orbitals = partial(get_orbitals, args=args)
    apply_fast = partial(apply_nnb_fast, args=args)

    def symm_apply(params, data, cache):
        return network_block.symmetrize_apply(params, data, cache, args, apply, apply_fast)

    return init, symm_apply, orbitals
