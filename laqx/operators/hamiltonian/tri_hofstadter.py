import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    def prepare(spin_up):
        operators = []
        weights = []
        for i in range(args.L1):
            for j in range(args.L2):
                start = args.L2 * i + j
                end1 = args.L2 * ((i + 1) % args.L1) + j
                end2 = args.L2 * i + ((j + 1) % args.L2)
                mul = 1 if j % 2 == 0 else -1

                if (i != args.L1 - 1) or (args.boundary1 == 'pbc'):
                    operators += [[start, end1], [end1, start]]
                    weights += [1j * mul, -1j * mul]
                if (j != args.L2 - 1) or (args.boundary2 == 'pbc'):
                    operators += [[start, end2], [end2, start]]
                    if args.flux_theta and (j == args.L2 // 2 - 1) and (i < args.L1 // 2):
                        weights += [-mul * np.exp(1j * args.flux_theta * spin_up * 2 * np.pi), -mul * np.exp(-1j * args.flux_theta * spin_up * 2 * np.pi)]
                    else:
                        weights += [-mul, -mul]
                if ((i != args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j != args.L2 - 1) or (args.boundary2 == 'pbc')):
                    operators += [[end1, end2], [end2, end1]]
                    if args.flux_theta and (j == args.L2 // 2 - 1) and (i < args.L1 // 2 - 1):
                        weights += [-1 * np.exp(1j * args.flux_theta * spin_up * 2 * np.pi), -1 * np.exp(-1j * args.flux_theta * spin_up * 2 * np.pi)]
                    else:
                        weights += [-1, -1]
                    

        operators = jnp.array(operators)
        weights = jnp.array(weights)
        return operators, weights
    
    operators_up, weights_up = prepare(1)
    flux_down = -1 if args.flux_type == 'spin' else 1
    operators_down, weights_down = prepare(flux_down)
    operators_down = operators_down + args.L1 * args.L2
    operators = jnp.concatenate([operators_up, operators_down], axis=0)
    weights = jnp.concatenate([weights_up, weights_down], axis=0)
    return operators, weights

def get_operators_and_weights_a(args):
    return None, None

def get_operators_and_weights_U(args):
    operators, weights = [], []
    for i in range(args.L1):
        for j in range(args.L2):
            operators += [[i * args.L2 + j, i * args.L2 + j + args.L1 * args.L2]]
            weights += [args.U]
    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights



def get_operators_and_weights_h(args):
    return None, None
    


def make_tri_hofstadter(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

