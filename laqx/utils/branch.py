import jax
import jax.numpy as jnp
import numpy as np

def branch(weight, key, data, min_thres=0.3, max_thres=2):
    num_merge_pairs = (jnp.sum(weight < min_thres) + 1) // 2
    num_split_walkers = jnp.sum(weight > max_thres)
    num_to_change = jnp.maximum(num_merge_pairs, num_split_walkers)
    if num_to_change == 0:
        return weight, data, 0

    _, smallest_k_indices = jax.lax.top_k(-1 * weight, 2 * num_to_change)
    smallest_k_indices = smallest_k_indices.reshape((num_to_change, 2))

    _, largest_k_indices = jax.lax.top_k(weight, num_to_change)
    weight, data = do_branch(weight, key, smallest_k_indices, largest_k_indices, data)

    return weight, data, num_to_change

@jax.jit
def do_branch(weight, key, smallest_k_indices, largest_k_indices, data):
    thresholds = weight[smallest_k_indices[:, 0]] / (weight[smallest_k_indices[:, 0]] + weight[smallest_k_indices[:, 1]])
    random_num = jax.random.uniform(key, shape=thresholds.shape)
    kept_indices = jnp.where(random_num < thresholds, smallest_k_indices[:, 0], smallest_k_indices[:, 1])
    removed_indices = jnp.where(random_num > thresholds, smallest_k_indices[:, 0], smallest_k_indices[:, 1])

    data = data.at[removed_indices].set(data[largest_k_indices])

    weight = weight.at[kept_indices].add(weight[removed_indices])
    weight = weight.at[largest_k_indices].set(weight[largest_k_indices] / 2)
    weight = weight.at[removed_indices].set(weight[largest_k_indices])
    return weight, data
