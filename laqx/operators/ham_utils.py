import jax
import jax.numpy as jnp
import numpy as np


def make_diag(operators, weights):
    if len(operators.shape) == 2:
        operators = operators[None]
        weights = weights[None]
        increase = True
    else:
        increase = False

    def _U(data):
        if operators.shape[-1] == 1:
            occupation = data[operators[:, :, 0]]
        elif operators.shape[-1] == 2:
            occupation = jnp.prod(data[operators], axis=-1)
        results = jnp.sum(occupation * weights, axis=-1)
        if increase:
            results = results[0]
        return results
    
    return jax.vmap(_U)

def make_non_diag(network, args, operators, weights):
    if operators.shape[-1] == 2:
        apply = apply_first
        reduce_number = args.reduce
    elif operators.shape[-1] == 4:
        apply = apply_second
        reduce_number = args.reduce2

    if len(operators.shape) == 2:
        operators = operators[None]
        weights = weights[None]
        increase = True
    else:
        increase = False

    def _t_fast(params, data, weight, cache):
        new_cache = network(params, data, cache)
        if args.num_states > 1:
            return weight * new_cache['psi'] / cache['psi']
        else:
            return weight * cache['sign'] * new_cache['sign'] * jnp.exp(new_cache['logdet'] - cache['logdet'])
    _t_fast = jax.vmap(_t_fast, in_axes=(None, 0, 0, 0))

    def _t_all_scan(params, data):
        init_network = lambda data: network(params, data, None)
        cache = jax.vmap(init_network)(data)
        
        if args.num_states == 1:
            cache['logdet'] = cache['logdet'].astype(jnp.float64) if args.dtype != 'complex' else cache['logdet'].astype(jnp.complex128)

        reduce_number_batch = data.shape[0] * reduce_number

        apply_sign_ori, new_data = jax.vmap(jax.vmap(jax.vmap(apply, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(0, None))(data, operators)
        if args.fast_update or args.use_boson:
            apply_sign_ori = jnp.abs(apply_sign_ori)
        apply_sign = apply_sign_ori * weights
        B, R, O = apply_sign.shape

        apply_log = jnp.sum(jnp.where(apply_sign != 0, 1, 0)) / B
        reduce = jnp.nonzero(apply_sign.reshape(-1), size=reduce_number_batch, fill_value=-1)[0]
        apply_sign = (apply_sign.reshape(-1)[reduce] * jnp.where(reduce != -1, 1, 0)).reshape(reduce_number, data.shape[0])
        new_data = new_data.reshape(-1, 2 * args.L1 * args.L2)[reduce].reshape(reduce_number, data.shape[0], 2 * args.L1 * args.L2)

        root = jnp.arange(B * R)
        root = jnp.tile(root[:, None], (1, O)).reshape(-1)
        root = root[reduce]

        reduce_index = root.reshape(reduce_number, B) // R

        if args.fast_update:
            def body_fn(carry, operator):
                slice_cache = jax.tree_map(lambda x: x[operator[2]], cache)
                result = _t_fast(params, operator[0], operator[1], slice_cache)
                return jnp.maximum(jnp.max(slice_cache['neighbor']), carry), result

        else:
            def body_fn(carry, operator):
                slice_cache = jax.tree_map(lambda x: x[operator[2]], cache)
                result = _t_fast(params, operator[0], operator[1], slice_cache)
                return 0, result

        max_neighbor, final_results = jax.lax.scan(body_fn, 0, (new_data, apply_sign, reduce_index))

        if args.num_states > 1:
            final_results = final_results.reshape(-1, args.num_states)
        else:
            final_results = final_results.reshape(-1)

        results = jax.ops.segment_sum(final_results, root, num_segments=B * R)
        if not increase:
            results = results.reshape(B, R)
        return results, {'apply': apply_log, 'neighbor': max_neighbor}
    
    return _t_all_scan



def check_first(data, operator):
    return data[operator[0]] * (1 - data[operator[1]])

def apply_first(data, operator):
    apply_sign = data[operator[0]]
    mask = jnp.arange(len(data)) < operator[0]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[0]].set(1 - data[operator[0]])

    apply_sign *= (1 - data[operator[1]])
    mask = jnp.arange(len(data)) < operator[1]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[1]].set(1 - data[operator[1]])
    return apply_sign, data

def check_second(data, operator):
    return data[operator[0]] * data[operator[1]] * (1 - data[operator[2]]) * (1 - data[operator[3]])

def apply_second(data, operator):
    apply_sign = data[operator[0]]
    mask = jnp.arange(len(data)) < operator[0]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[0]].set(1 - data[operator[0]])

    apply_sign *= data[operator[1]]
    mask = jnp.arange(len(data)) < operator[1]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[1]].set(1 - data[operator[1]])

    apply_sign *= (1 - data[operator[2]])
    mask = jnp.arange(len(data)) < operator[2]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[2]].set(1 - data[operator[2]])

    apply_sign *= (1 - data[operator[3]])
    mask = jnp.arange(len(data)) < operator[3]
    apply_sign *= (-1) ** jnp.sum(data * mask)
    data = data.at[operator[3]].set(1 - data[operator[3]])
    return apply_sign, data