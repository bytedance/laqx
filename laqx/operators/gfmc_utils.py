import jax
import jax.numpy as jnp
from functools import partial
from laqx.operators import ham_utils

def make_Heff(network, operator, args):
    t_operator, t_weights = operator['t']
    a_operator, a_weights = operator['a']
    U_operator, U_weights = operator['U']
    h_operator, h_weights = operator['h']
    if U_operator is not None:
        U_apply = ham_utils.make_diag(U_operator, U_weights)
    else:
        U_apply = None
    if h_operator is not None:
        h_apply = ham_utils.make_diag(h_operator, h_weights)
    else:
        h_apply = None

    def _t_fast(params, data, weight, cache):
        new_cache = network(params, data, cache)
        return weight * cache['sign'] * new_cache['sign'] * jnp.exp(new_cache['logdet'] - cache['logdet'])
    
    def evaluate_Heff(params, data):
        cache = network(params, data, None)
        cache['logdet'] = cache['logdet'].astype(jnp.float64) if args.dtype != 'complex' else cache['logdet'].astype(jnp.complex128)

        def prepare(check, O, W, R):
            apply_sign_ori, new_data = jax.vmap(check, in_axes=(None, 0))(data, O)
            if args.fast_update or args.use_boson:
                apply_sign_ori = jnp.abs(apply_sign_ori)
            apply_sign = apply_sign_ori * W
            
            reduce = jnp.nonzero(apply_sign, size=R, fill_value=-1)[0]

            apply_sign = apply_sign[reduce] * jnp.where(reduce != -1, 1, 0)
            new_data = new_data[reduce]
            return apply_sign, new_data
        
        if t_operator is not None:
            apply_sign_t, new_data_t = prepare(ham_utils.apply_first, t_operator, t_weights, args.reduce)
        if a_operator is not None:
            apply_sign_a, new_data_a = prepare(ham_utils.apply_second, a_operator, a_weights, args.reduce2)
        
        if args.fast_update:
            def body_fn(carry, operator):
                slice_cache = jax.tree_map(lambda x: x.copy(), cache)
                result = _t_fast(params, operator[0], operator[1], slice_cache)
                return jnp.maximum(slice_cache['neighbor'], carry), result

        else:
            def body_fn(carry, operator):
                result = _t_fast(params, operator[0], operator[1], cache)
                return 0, result

        max_neighbor = 0
        if t_operator is not None:
            max_neighbor, results_t = jax.lax.scan(body_fn, max_neighbor, (new_data_t, apply_sign_t))
            SF_t = jnp.where(results_t > 0, 1, 0)
        if a_operator is not None:
            max_neighbor, results_a = jax.lax.scan(body_fn, max_neighbor, (new_data_a, apply_sign_a))
            SF_a = jnp.where(results_a > 0, 1, 0)
        
        H_diag = 0
        if U_apply is not None:
            H_diag += U_apply(data[None])[0]
        if h_apply is not None:
            H_diag += h_apply(data[None])[0]
        if t_operator is not None:
            H_diag += jnp.sum(results_t * SF_t)
        if a_operator is not None:
            H_diag += jnp.sum(results_a * SF_a)

        if t_operator is not None and a_operator is not None:
            results = jnp.concatenate([results_t * (1 - SF_t), results_a * (1 - SF_a)])
            new_data = jnp.concatenate([new_data_t, new_data_a])
        elif t_operator is not None:
            results = results_t * (1 - SF_t)
            new_data = new_data_t
        elif a_operator is not None:
            results = results_a * (1 - SF_a)
            new_data = new_data_a
    
        return H_diag, results, new_data, max_neighbor

    return evaluate_Heff


def make_green_function(network, operators, args):
    evaluate_Heff = make_Heff(network, operators, args)

    def step(params, data, weight, Et, key):
        H_diag, H_nondiag, new_data, neighbor = evaluate_Heff(params, data)
        e_l = H_diag + jnp.sum(H_nondiag)
        population = H_diag - Et
        gfmc_step = jnp.where(population > 1 / args.gfmc_step, 1 / population, args.gfmc_step)
        prob = jnp.concatenate([1 - gfmc_step * population[None], -gfmc_step * H_nondiag])
        data_update = jax.random.choice(key, jnp.concatenate([data[None], new_data]), p=prob)
        weight = weight * (1 + gfmc_step * (Et - e_l))
        aux = {'gfmc_step': gfmc_step, 'neighbor': neighbor}
        return data_update, weight, e_l, aux
    
    return step
    
def get_gfmc_update_fn(network, operators, args):
    green_step = make_green_function(network, operators, args)
    batch_green_step = jax.vmap(green_step, in_axes=(None, 0, 0, None, 0))

    def gfmc_update_fn(params, data, weights, Et, key):
        data_update, weight_update, e_l, aux = batch_green_step(params, data, weights, Et, key)
        total_weight = jax.lax.psum(jnp.sum(weights), axis_name='batch')
        loss = jax.lax.psum(jnp.sum(weights * e_l), axis_name='batch') / total_weight
        logdict = {'loss': loss, 'offset': Et, 'mean_step': jax.lax.pmean(jnp.mean(aux['gfmc_step']), axis_name='batch'), \
            'neighbor': jax.lax.pmax(jnp.max(aux['neighbor']), axis_name='batch')}

        logdict = jax.tree_map(lambda x: x.astype(jnp.float32), logdict)
        Et = loss
        weight_update = weight_update / jax.lax.pmean(jnp.mean(weight_update), axis_name='batch')
        return data_update, weight_update, Et, logdict

    return gfmc_update_fn