import jax
import jax.numpy as jnp

from laqx import operators

def smooth_solve(S, F, pinv_tol=1e-14, pinv_cutoff=1e-8):
    ein_val, V = jnp.linalg.eigh(S)
    VtF = jnp.dot(jnp.transpose(jnp.conj(V)), F)
    inv_ein_val = jnp.where(jnp.abs(ein_val / ein_val[-1]) > pinv_tol, 1. / ein_val, 0.)
    pinv_ein_val = inv_ein_val / (1. + (pinv_cutoff / jnp.abs(ein_val / ein_val[-1]))**6)
    update = jnp.dot(V, (pinv_ein_val * VtF))
    return update

def get_tdvp_update_fn(network, local_energy, args, batch_mcmc_step, num_devices):
    log_network = lambda params, data: network(params, data, None)['logdet']
    log_network_wrapper = lambda params_peft, params_fixed, data: log_network({'peft': params_peft, 'fixed': params_fixed}, data)
    grad_network_holomorphic = jax.grad(log_network_wrapper, holomorphic=True)
    grad_network = lambda params, data: grad_network_holomorphic(params['peft'], params['fixed'], data)

    def raveled_log_psi_grad(params, data):
        log_grads = grad_network(params, data)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def cal_update(params, prev_grad, data, e_l, t, key):
        param_flat, unravel_fn = jax.flatten_util.ravel_pytree(params['peft'])
        log_psi_grads = batch_raveled_log_psi_grad(params, data) / jnp.sqrt(args.batchsize)
        
        Ohat = log_psi_grads - jax.lax.pmean(jnp.mean(log_psi_grads, axis=0, keepdims=True), axis_name='batch')
        Ohat_original = Ohat

        diff = e_l * 1j
        diff = diff - jax.lax.pmean(jnp.mean(diff), axis_name='batch')
        np = Ohat.shape[-1]

        diff = -diff / jnp.sqrt(args.batchsize)
        epsilon_tilde = diff

        T = jax.lax.psum(Ohat.T.conj() @ Ohat, axis_name='batch')
        B = T.shape[0]
        epsilon_tilde = jax.lax.psum(Ohat.T.conj() @ epsilon_tilde, axis_name='batch')
        
        if args.solver == 'cholesky':
            dtheta_residual = jax.scipy.linalg.solve(T + args.damping * jnp.eye(B), epsilon_tilde, assume_a="pos")
        else:
            dtheta_residual = smooth_solve(T, epsilon_tilde, pinv_cutoff=args.pinv_cutoff)

        grad = dtheta_residual

        # Check SR residual
        res_error = diff - Ohat_original @ grad
        res_error = jax.lax.pmean(jnp.mean(jnp.abs(res_error)**2), axis_name='batch')
        res_norm = jax.lax.pmean(jnp.mean(jnp.abs(diff)**2), axis_name='batch')
        res = res_error / res_norm

        grad_after_norm = grad * args.lr
        return data, unravel_fn(param_flat + grad_after_norm), grad, (jnp.linalg.norm(grad), res)

    def tdvp_update_fn(params, prev_grad, data, t, subkeys):
        if args.obs:
            f_obs = operators.operator_provider(network, args, inference=True)

        if args.integrator == 'rk2':
            data, aux_mcmc = batch_mcmc_step(params, data, subkeys)
            e_l1, aux_local = local_energy(params, data, {'cache': aux_mcmc['cache'], 't': t})
            if args.obs:
                obs1, _ = f_obs(params, data, {'cache': aux_mcmc['cache']})
            data, params2_peft, grad1, aux1 = cal_update(params, prev_grad, data, e_l1, t, aux_mcmc['key'])
            params2 = {'fixed': params['fixed'], 'peft': params2_peft}

            data, aux_mcmc = batch_mcmc_step(params2, data, aux_mcmc['key'])
            e_l2, aux_local2 = local_energy(params2, data, {'cache': aux_mcmc['cache'], 't': t+1})
            if args.obs:
                obs2, _ = f_obs(params2, data, {'cache': aux_mcmc['cache']})
            data, _, grad2, aux2 = cal_update(params2, prev_grad, data, e_l2, t, aux_mcmc['key'])

            param_flat, unravel_fn = jax.flatten_util.ravel_pytree(params['peft'])
            params_peft = unravel_fn(param_flat + (grad1 + grad2) * args.lr / 2)

            aux_local = {'apply': jnp.maximum(aux_local['apply'], aux_local2['apply']), 'neighbor': jnp.maximum(aux_local['neighbor'], aux_local2['neighbor'])}
            aux = ((aux1[0] + aux2[0]) / 2, (aux1[1] + aux2[1]) / 2)
            if args.obs:
                obs = (obs1 + obs2) / 2
            e_l = (e_l1 + e_l2) / 2
            opt_state = (grad1 + grad2) / 2

        elif args.integrator == 'rk4':
            param_flat, unravel_fn = jax.flatten_util.ravel_pytree(params['peft'])
            
            data, aux_mcmc1 = batch_mcmc_step(params, data, subkeys)
            e_l1, aux_local1 = local_energy(params, data, {'cache': aux_mcmc1['cache'], 't': t})
            if args.obs:
                obs1, _ = f_obs(params, data, {'cache': aux_mcmc1['cache']})
            data, _, grad1, aux1 = cal_update(params, prev_grad, data, e_l1, t, aux_mcmc1['key'])
            
            params2_peft = unravel_fn(param_flat + grad1 * (args.lr / 2.0))
            params2 = {'fixed': params['fixed'], 'peft': params2_peft}
            
            data, aux_mcmc2 = batch_mcmc_step(params2, data, aux_mcmc1['key'])
            e_l2, aux_local2 = local_energy(params2, data, {'cache': aux_mcmc2['cache'], 't': t + 0.5})
            if args.obs:
                obs2, _ = f_obs(params2, data, {'cache': aux_mcmc2['cache']})
            data, _, grad2, aux2 = cal_update(params2, prev_grad, data, e_l2, t + 0.5, aux_mcmc2['key'])

            params3_peft = unravel_fn(param_flat + grad2 * (args.lr / 2.0))
            params3 = {'fixed': params['fixed'], 'peft': params3_peft}
            
            data, aux_mcmc3 = batch_mcmc_step(params3, data, aux_mcmc2['key'])
            e_l3, aux_local3 = local_energy(params3, data, {'cache': aux_mcmc3['cache'], 't': t + 0.5})
            if args.obs:
                obs3, _ = f_obs(params3, data, {'cache': aux_mcmc3['cache']})
            data, _, grad3, aux3 = cal_update(params3, prev_grad, data, e_l3, t + 0.5, aux_mcmc3['key'])

            params4_peft = unravel_fn(param_flat + grad3 * args.lr)
            params4 = {'fixed': params['fixed'], 'peft': params4_peft}
            
            data, aux_mcmc4 = batch_mcmc_step(params4, data, aux_mcmc3['key'])
            e_l4, aux_local4 = local_energy(params4, data, {'cache': aux_mcmc4['cache'], 't': t + 1.0})
            if args.obs:
                obs4, _ = f_obs(params4, data, {'cache': aux_mcmc4['cache']})
            data, _, grad4, aux4 = cal_update(params4, prev_grad, data, e_l4, t + 1.0, aux_mcmc4['key'])

            params_peft = unravel_fn(param_flat + (grad1 + 2.0 * grad2 + 2.0 * grad3 + grad4) * (args.lr / 6.0))
            aux_local = {
                'apply': jnp.maximum(jnp.maximum(aux_local1['apply'], aux_local2['apply']), jnp.maximum(aux_local3['apply'], aux_local4['apply'])), 
                'neighbor': jnp.maximum(jnp.maximum(aux_local1['neighbor'], aux_local2['neighbor']), jnp.maximum(aux_local3['neighbor'], aux_local4['neighbor']))
            }
            aux = (
                (aux1[0] + 2.0 * aux2[0] + 2.0 * aux3[0] + aux4[0]) / 6.0, 
                (aux1[1] + 2.0 * aux2[1] + 2.0 * aux3[1] + aux4[1]) / 6.0
            )
            if args.obs:
                obs = (obs1 + 2.0 * obs2 + 2.0 * obs3 + obs4) / 6.0
            
            e_l = (e_l1 + 2.0 * e_l2 + 2.0 * e_l3 + e_l4) / 6.0
            opt_state = (grad1 + 2.0 * grad2 + 2.0 * grad3 + grad4) / 6.0
            aux_mcmc = aux_mcmc4    
        else:
            data, aux_mcmc = batch_mcmc_step(params, data, subkeys)
            e_l, aux_local = local_energy(params, data, {'cache': aux_mcmc['cache'], 't': t})
            if args.obs:
                obs, _ = f_obs(params, data, {'cache': aux_mcmc['cache']})
            data, params_peft, opt_state, aux = cal_update(params, prev_grad, data, e_l, t, aux_mcmc['key'])

        logdict = {'loss': jnp.mean(e_l), 'pmove': jnp.mean(aux_mcmc['pmove']), 'variance': jnp.var(jax.lax.all_gather(e_l, axis_name='batch')), \
                   'norm': aux[0], 'res': aux[1]}
        aux_local = jax.tree_map(lambda x: jax.lax.pmax(x, axis_name='batch'), aux_local)
        logdict = logdict | aux_local
        logdict_new = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch').astype(jnp.float32), logdict)
        if args.dtype == 'complex':
            logdict_new['loss'] = jax.lax.pmean(logdict['loss'], axis_name='batch').astype(jnp.complex64)
        if args.obs:
            logdict_new = (logdict_new, jnp.mean(obs, axis=0))
        return {'fixed': params['fixed'], 'peft': params_peft}, opt_state, data, logdict_new

    return tdvp_update_fn
