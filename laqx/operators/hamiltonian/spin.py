import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    sign = -1 if args.marshall else 1
    operators = []
    weights = []
    for i in range(args.L1):
        for j in range(args.L2):
            start = args.L2 * i + j
            end1 = args.L2 * ((i + 1) % args.L1) + j
            end2 = args.L2 * i + ((j + 1) % args.L2)
            if (i != args.L1 - 1) or (args.boundary1 == 'pbc'):
                operators += [[start, end1], [end1, start]]
                weights += [0.5 * args.j1 * sign, 0.5 * args.j1 * sign]
            if (j != args.L2 - 1) or (args.boundary2 == 'pbc'):
                operators += [[start, end2], [end2, start]]
                weights += [0.5 * args.j1 * sign, 0.5 * args.j1 * sign]

    if args.j2:
        for i in range(args.L1):
            for j in range(args.L2):
                start = args.L2 * i + j
                end1 = args.L2 * ((i + 1) % args.L1) + j
                end2 = args.L2 * i + ((j + 1) % args.L2)
                end3 = args.L2 * ((i + 1) % args.L1) + ((j + 1) % args.L2)
                if ((i < args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j < args.L2 - 1) or (args.boundary2 == 'pbc')):
                    operators += [[start, end3], [end3, start]]
                    weights += [0.5 * args.j2, 0.5 * args.j2]
                    operators += [[end1, end2], [end2, end1]]
                    weights += [0.5 * args.j2, 0.5 * args.j2]

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
                weights += [args.j1]
        
            end2 = args.L2 * i + ((j + 1) % args.L2)
            if (j < args.L2 - 1) or (args.boundary2 == 'pbc'):
                operators += [[start, end2]]
                weights += [args.j1]
    if args.j2:
        for i in range(args.L1):
            for j in range(args.L2):
                start = args.L2 * i + j
                end1 = args.L2 * ((i + 1) % args.L1) + j
                end2 = args.L2 * i + ((j + 1) % args.L2)
                end3 = args.L2 * ((i + 1) % args.L1) + ((j + 1) % args.L2)
                if ((i < args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j < args.L2 - 1) or (args.boundary2 == 'pbc')):
                    operators += [[start, end3]]
                    weights += [args.j2]
                    operators += [[end1, end2]]
                    weights += [args.j2]

    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights



def get_operators_and_weights_h(args):
    operators, weights = [], []
    for i in range(args.L1):
        for j in range(args.L2):
            start = args.L2 * i + j
            end1 = args.L2 * ((i + 1) % args.L1) + j
            if (i < args.L1 - 1) or (args.boundary1 == 'pbc'):
                operators += [[start], [end1]]
                weights += [-args.j1 / 2, -args.j1 / 2]
        
            end2 = args.L2 * i + ((j + 1) % args.L2)
            if (j < args.L2 - 1) or (args.boundary2 == 'pbc'):
                operators += [[start], [end2]]
                weights += [-args.j1 / 2, -args.j1 / 2]
    
    if args.j2:
        for i in range(args.L1):
            for j in range(args.L2):
                start = args.L2 * i + j
                end1 = args.L2 * ((i + 1) % args.L1) + j
                end2 = args.L2 * i + ((j + 1) % args.L2)
                end3 = args.L2 * ((i + 1) % args.L1) + ((j + 1) % args.L2)
                if ((i < args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j < args.L2 - 1) or (args.boundary2 == 'pbc')):
                    operators += [[start], [end3]]
                    weights += [-args.j2 / 2, -args.j2 / 2]
                    operators += [[end1], [end2]]
                    weights += [-args.j2 / 2, -args.j2 / 2]

    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights
    


def make_spin(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

