import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    def get_operator_weight():
        operators = []
        weights = []
        for i in range(args.L1):
            for j in range(args.L2):
                start = args.L2 * i + j
                end1 = args.L2 * ((i + 1) % args.L1) + j
                end2 = args.L2 * i + ((j + 1) % args.L2)

                if (i < args.L1 - 1) or (args.boundary1 == 'pbc'):
                    operators += [[start, end1], [end1, start]]
                    weights += [-args.t, -args.t]
                if (j < args.L2 - 1) or (args.boundary2 == 'pbc'):
                    operators += [[start, end2], [end2, start]]
                    weights += [-args.t, -args.t]

        if args.t2 != 0:
            for i in range(args.L1):
                for j in range(args.L2):
                    if (i + j) % 2 == 0:
                        start = args.L2 * i + j
                        end1 = args.L2 * ((i + 1) % args.L1) + j
                        end2 = args.L2 * i + ((j + 1) % args.L2)
                        end3 = args.L2 * ((i + 1) % args.L1) + ((j + 1) % args.L2)

                        if ((i < args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j < args.L2 - 1) or (args.boundary2 == 'pbc')):
                            operators += [[start, end3], [end3, start], [end1, end2], [end2, end1]]
                            weights += [-args.t2, -args.t2, -args.t2, -args.t2]                    


        operators = jnp.array(operators)
        weights = jnp.array(weights)
        return operators, weights
    
    operators_up, weights_up = get_operator_weight()
    operators_down, weights_down = operators_up, weights_up
    operators = jnp.concatenate([operators_up, operators_down + args.L1 * args.L2], axis=0)
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

def make_hubbard_alter(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

