import jax
import jax.numpy as jnp
import numpy as np

def get_operators_and_weights_t(args):
    def get_operator_weight(spin):
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
                    start = args.L2 * i + j
                    end1 = args.L2 * ((i + 1) % args.L1) + j
                    end2 = args.L2 * i + ((j + 1) % args.L2)
                    end3 = args.L2 * ((i + 1) % args.L1) + ((j + 1) % args.L2)

                    if ((i < args.L1 - 1) or (args.boundary1 == 'pbc')) and ((j < args.L2 - 1) or (args.boundary2 == 'pbc')):
                        operators += [[start, end3], [end3, start], [end1, end2], [end2, end1]]
                        t2 = -args.t2 * spin
                        weights += [t2, t2, t2, t2]

        if args.t3 != 0:
            for i in range(args.L1):
                for j in range(args.L2):
                    start = args.L2 * i + j
                    end1 = args.L2 * ((i + 2) % args.L1) + j
                    end2 = args.L2 * i + ((j + 2) % args.L2)

                    if (i < args.L1 - 2) or (args.boundary1 == 'pbc'):
                        operators += [[start, end1], [end1, start]]
                        weights += [-args.t3, -args.t3]
                    if (j < args.L2 - 2) or (args.boundary2 == 'pbc'):
                        operators += [[start, end2], [end2, start]]
                        weights += [-args.t3, -args.t3] 

        operators = jnp.array(operators)
        weights = jnp.array(weights)
        return operators, weights
    
    operators_up, weights_up = get_operator_weight(1)
    if args.particle_hole:
        operators_down, weights_down = get_operator_weight(-1)
    else:
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
    if args.hm == 0:
        return None, None
    all_operators = np.zeros((args.L1, args.L2))
    if 'hori' in args.htype:
        all_operators = all_operators.T

    all_operators_spin = np.zeros_like(all_operators)
    all_operators_hole = np.zeros_like(all_operators)

    if 'neel' in args.htype:
        all_operators_spin = np.copy(all_operators)
        for i in range(all_operators_spin.shape[0]):
            all_operators_spin[i] = (-1) ** (np.arange(all_operators_spin.shape[1]) + i)
        if 'hori' in args.htype:
            all_operators_spin = all_operators_spin.T

    if 'spin' in args.htype:
        all_operators_spin = np.copy(all_operators)
        for i in range(all_operators_spin.shape[0]):
            if 'left' in args.htype and i >= 1:
                break
            if 'obc' in args.htype:
                if (i != 0) and (i != all_operators_spin.shape[0] - 1):
                    continue 
            if (i % args.lambda_h == 0) or ((i + 1) % args.lambda_h == 0):
                all_operators_spin[i] = (-1) ** (np.arange(all_operators_spin.shape[1]) + (i + 1) // args.lambda_h + i)
        if 'hori' in args.htype:
            all_operators_spin = all_operators_spin.T

    if 'AFM' in args.htype:
        all_operators_spin = np.copy(all_operators)
        for i in range(all_operators_spin.shape[0]):
            if (i % args.lambda_h == 0) or ((i + 1) % args.lambda_h == 0):
                all_operators_spin[i] = (-1) ** (np.arange(all_operators_spin.shape[1]) + i)
        if 'hori' in args.htype:
            all_operators_spin = all_operators_spin.T

    if 'board' in args.htype:
        all_operators_hole = np.copy(all_operators)
        for i in range(all_operators_hole.shape[0]):
            all_operators_hole[i] = (-1) ** (np.arange(all_operators_hole.shape[1]) + i)
        if 'hori' in args.htype:
            all_operators_hole = all_operators_hole.T
    
    if 'hole' in args.htype:
        all_operators_hole = np.copy(all_operators)
        for i in range(all_operators_hole.shape[0]):
            if i % args.lambda_h == 0:
                all_operators_hole[i] = 1
            if (i + 1) % args.lambda_h == 0:
                all_operators_hole[i] = 1
            if (i % args.lambda_h) == ((args.lambda_h + 1) // 2 - 1):
                all_operators_hole[i] = -1
            if (i % args.lambda_h) == (args.lambda_h // 2):
                all_operators_hole[i] = -1
        if 'hori' in args.htype:
            all_operators_hole = all_operators_hole.T

    operators, weights = [], []
    has_spin = any(k in args.htype for k in ['spin', 'neel', 'AFM'])
    has_hole = any(k in args.htype for k in ['hole', 'board'])

    for i in range(args.L1):
        for j in range(args.L2):
            idx_up = i * args.L2 + j
            idx_dn = i * args.L2 + j + args.L1 * args.L2
            
            w_spin = all_operators_spin[i, j] if has_spin else 0.0
            w_hole = all_operators_hole[i, j] if has_hole else 0.0
            
            w_up = 0.0
            w_dn = 0.0
            
            if has_spin:
                if not args.particle_hole:
                    w_up += w_spin / 2.0
                    w_dn -= w_spin / 2.0
                else:
                    w_up += w_spin / 2.0
                    w_dn += w_spin / 2.0
                    
            if has_hole:
                if not args.particle_hole:
                    w_up -= 3.0 * w_hole
                    w_dn -= 3.0 * w_hole
                else:
                    w_up -= 3.0 * w_hole
                    w_dn += 3.0 * w_hole
                    
            if w_up != 0:
                operators.append([idx_up])
                weights.append(w_up)
            if w_dn != 0:
                operators.append([idx_dn])
                weights.append(w_dn)

    operators = jnp.array(operators)
    weights = jnp.array(weights) * args.hm
    return operators, weights


def make_hubbard(args):
    return {'t': get_operators_and_weights_t(args), 'a': get_operators_and_weights_a(args), 'U': get_operators_and_weights_U(args), 'h': get_operators_and_weights_h(args)}

