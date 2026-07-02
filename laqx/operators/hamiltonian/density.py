import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    return None, None

def get_operators_and_weights_a(args):
    return None, None

def get_operators_and_weights_U(args):
    return None, None

def get_operators_and_weights_h(args):
    operators, weights = [], []
    for i in range(args.L1):
        for j in range(args.L2):
            operators.append([[i * args.L2 + j], [i * args.L2 + j + args.L1 * args.L2]])
            weights.append([1, 1])

    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights

def make_density(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

