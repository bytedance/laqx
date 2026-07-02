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

                phase = 2 * np.pi * (args.flux_theta / args.L1)

                if (i != args.L1 - 1) or (args.boundary1 == 'pbc'):
                    operators += [[start, end1], [end1, start]]
                    weights += [args.t * np.exp(-1j * phase), args.t * np.exp(1j * phase)]
                if (i % 2) == 0:
                    end2 = args.L2 * ((i - 1) % args.L1) + ((j + 1) % args.L2)
                    operators += [[start, end2], [end2, start]]
                    weights += [args.t * np.exp(1j * phase), args.t * np.exp(-1j * phase)]

        if args.t2:
            for i in range(args.L1):
                for j in range(args.L2):
                    parity = i % 2
                    R = (i + 1, j)
                    L = (i - 1, j)
                    if parity == 0:
                        U = (i - 1, j + 1)
                        neighbors = {'R': R, 'L': L, 'U': U}
                        left_turns = [('U', 'R'), ('R', 'L'), ('L', 'U')]
                    else:          
                        D = (i + 1, j - 1)
                        neighbors = {'R': R, 'L': L, 'D': D}
                        left_turns = [('R', 'D'), ('D', 'L'), ('L', 'R')]

                    for start, end in left_turns:
                        n_start = neighbors[start]
                        n_end = neighbors[end]  
                        j1 = (n_start[0] % args.L1) * args.L2 + (n_start[1] % args.L2)
                        j2 = (n_end[0] % args.L1) * args.L2 + (n_end[1] % args.L2)

                        dx_total = n_end[0] - n_start[0]
                        phase_flux = 2 * np.pi * (args.flux_theta / args.L1) * dx_total
                        amp = args.t2 * np.exp(-1j * phase_flux) * np.exp(1j * 2 * np.pi / 3)

                        operators += [[j1, j2], [j2, j1]]
                        weights += [amp, np.conj(amp)]

        if args.t3:
            for i in range(args.L1):
                for j in range(args.L2):
                    if (i % 2) != 0:
                        continue
                    start = args.L2 * i + j
                    t3_targets = [
                        (i + 1, j - 1),
                        (i + 1, j + 1),  
                        (i - 3, j + 1),
                    ]

                    for x_end_raw, y_end_raw in t3_targets:
                        end = (x_end_raw % args.L1) * args.L2 + (y_end_raw % args.L2)
                        dx_total = x_end_raw - i
                        
                        phase_flux_t3 = 2 * np.pi * (args.flux_theta / args.L1) * dx_total
                        amp_t3 = args.t3 * np.exp(-1j * phase_flux_t3)

                        operators += [[start, end], [end, start]]
                        weights += [amp_t3, np.conj(amp_t3)]

                    

        operators = jnp.array(operators)
        weights = jnp.array(weights)
        return operators, weights
    
    operators_up, weights_up = prepare(1)
    return operators_up, weights_up

def get_operators_and_weights_a(args):
    return None, None

def get_operators_and_weights_U(args):
    operators, weights = [], []
    for i in range(args.L1):
        for j in range(args.L2):
            start = args.L2 * i + j
            end1 = args.L2 * ((i + 1) % args.L1) + j
            operators += [[start, end1]]
            weights += [args.V]
        
            end2 = args.L2 * ((i - 1) % args.L1) + ((j + 1) % args.L2)
            if (i % 2 == 0):
                operators += [[start, end2]]
                weights += [args.V]
    operators = jnp.array(operators)
    weights = jnp.array(weights)
    return operators, weights



def get_operators_and_weights_h(args):
    return None, None
    


def make_haldane(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

