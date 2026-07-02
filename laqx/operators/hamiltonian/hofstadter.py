import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    operators = []
    weights = []
    for i in range(args.L1):
        for j in range(args.L2):
            start = args.L2 * i + j
            end1 = args.L2 * ((i + 1) % args.L1) + j
            end2 = args.L2 * i + ((j + 1) % args.L2)

            if (i != args.L1 - 1) or (args.boundary1 == 'pbc'):
                operators += [[start, end1], [end1, start]]
                phase = 2 * np.pi * (args.flux_theta / args.L1)
                weights += [-1 * np.exp(-1j * phase), -1 * np.exp(1j * phase)]
            if (j != args.L2 - 1) or (args.boundary2 == 'pbc'):
                operators += [[start, end2], [end2, start]]
                phase = 2 * np.pi * (args.alpha * i)
                weights += [-1 * np.exp(-1j * phase), -1 * np.exp(1j * phase)]

    operators = jnp.array(operators)
    weights = jnp.array(weights)
    
    return operators, weights

def get_operators_and_weights_a(args):
    return None, None

def get_operators_and_weights_U(args):
    operators, weights = [], []
    for i in range(args.L1):
        for j in range(args.L2):
            start = args.L2 * i + j
            end1 = args.L2 * ((i + 1) % args.L1) + j
            if (i < args.L1 - 1) or (args.boundary1 == 'pbc'):
                operators += [[start, end1]]
                weights += [args.V]
        
            end2 = args.L2 * i + ((j + 1) % args.L2)
            if (j < args.L2 - 1) or (args.boundary2 == 'pbc'):
                operators += [[start, end2]]
                weights += [args.V]
    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights



def get_operators_and_weights_h(args):
    if args.hv == 0:
        return None, None
    operators = jnp.array([[0]])
    weights = jnp.array([args.hv])
    return operators, weights
    


def make_hofstadter(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

