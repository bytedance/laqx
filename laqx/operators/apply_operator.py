import jax
import jax.numpy as jnp
import numpy as np
from laqx.operators import ham_utils

def make_nes_diag(func):
    def _func_nes(data):
        results = jax.vmap(func)(data)
        results = jnp.broadcast_to(results[:, :, None], (results.shape[0], results.shape[1], results.shape[1]))
        return results
    return _func_nes

def make_nes_non_diag(func, args):
    def _func_nes(params, data):
        results, log = func(params, data.reshape(-1, data.shape[-1]))
        results = results.reshape(-1, args.num_states, args.num_states)
        return results, log
    return _func_nes

def make_operator(network, operator, args):
    t_operator, t_weights = operator['t']
    a_operator, a_weights = operator['a']
    U_operator, U_weights = operator['U']
    h_operator, h_weights = operator['h']
    if t_operator is not None:
        t_apply = ham_utils.make_non_diag(network, args, t_operator, t_weights)
        if args.num_states > 1:
            t_apply = make_nes_non_diag(t_apply, args)
    else:
        t_apply = None
    
    if a_operator is not None:
        a_apply = ham_utils.make_non_diag(network, args, a_operator, a_weights)
        if args.num_states > 1:
            a_apply = make_nes_non_diag(a_apply, args)
    else:
        a_apply = None

    if U_operator is not None:
        U_apply = ham_utils.make_diag(U_operator, U_weights)
        if args.num_states > 1:
            U_apply = make_nes_diag(U_apply)
    else:
        U_apply = None

    if h_operator is not None:
        h_apply = ham_utils.make_diag(h_operator, h_weights)
        if args.num_states > 1:
            h_apply = make_nes_diag(h_apply)
    else:
        h_apply = None

    def _e_l(params, data, aux):
        if t_apply is not None:
            et, log_t = t_apply(params, data)
        else:
            et, log_t = 0, {}
        
        if a_apply is not None:
            ea, log_a = a_apply(params, data)
            if log_t:
                log_t['neighbor'] = jnp.maximum(log_t['neighbor'], log_a['neighbor'])
                log_t['apply2'] = log_a['apply']
            else:
                log_t = log_a
        else:
            ea, log_a = 0, {}

        if args.fast_update:
            if aux is not None and aux.get('cache') is not None and 'neighbor' in aux['cache']:
                previous_neighbor = jnp.max(aux['cache']['neighbor'])
            else:
                previous_neighbor = jnp.array(0)
            if log_t:
                log_t['neighbor'] = jnp.maximum(log_t['neighbor'], previous_neighbor)
            else:
                log_t = {'neighbor': previous_neighbor}

        if U_apply is not None:
            eu = U_apply(data)
        else:
            eu = 0

        if h_apply is not None:
            eh = h_apply(data)
        else:
            eh = 0
        
        if args.particle_hole:
            spin_up = jnp.sum(data.reshape(-1, 2, args.L1 * args.L2)[:, 0], axis=1)
            log_t['spin_up'] = jax.lax.pmean(jnp.mean(spin_up), axis_name='batch')

            spin_delta = spin_up - args.particles_up
            deviation_thresholds = jnp.arange(5) + 1
            def count_deviation(t):
                return jnp.sum(jnp.abs(spin_delta) >= t) 
            counts = jax.vmap(count_deviation)(deviation_thresholds)
            log_t['spin_other'] = jax.lax.psum(counts, axis_name='batch')

        if args.num_states == 1:
            return et + ea + eu + eh, log_t
        else:
            psi = jax.vmap(jax.vmap(network, in_axes=(None, 0, None)), in_axes=(None, 0, None))(params, data, None)['psi']
            el = et + ea + eu + eh
            energy_matrix = jnp.matmul(jnp.linalg.inv(psi), el * psi)
            return energy_matrix, log_t

    return _e_l
