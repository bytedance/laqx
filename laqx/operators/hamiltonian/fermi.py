import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    operators = []
    weights = []
    for r1 in range(args.L1):
        for r2 in range(args.L2):
            operator = []
            weight = []
            for i in range(0, 4):
                for j in range(0, 4):
                    start = args.L2 * ((i + r1) % args.L1) + (j + r2) % args.L2
                    end = args.L2 * i + j
                    operator.append([start, end])
                    weight.append(1)

            operators.append(operator)
            weights.append(weight)

    operators = jnp.array(operators)
    operators = jnp.concatenate([operators, operators + args.L1 * args.L2], axis=0)
    weights = jnp.array(weights) / operators.shape[1]
    weights = jnp.concatenate([weights, weights], axis=0)
    return operators, weights

def get_operators_and_weights_a(args):
    return None, None

def get_operators_and_weights_U(args):
    return None, None

def get_operators_and_weights_h(args):
    return None, None

def make_fermi(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

