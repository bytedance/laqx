import jax
import jax.numpy as jnp

def get_spring_update_fn(network, local_energy, args, batch_mcmc_step, num_devices):
    if args.num_states == 1:
        log_network = lambda params, data: network(params, data, None)['logdet']
    else:
        def log_network(params, data):
            psi = jax.vmap(network, in_axes=(None, 0, None))(params, data, None)['psi']
            sign, logdet = jnp.linalg.slogdet(psi)
            if args.dtype == 'complex':
                logdet = jnp.log(sign) + logdet
            return logdet
    if args.dtype != 'complex':
        grad_network = jax.grad(log_network)
    else:
        log_network_real = lambda params, data: jnp.real(log_network(params, data))
        log_network_imag = lambda params, data: jnp.imag(log_network(params, data))
        grad_network_real = jax.grad(log_network_real)
        grad_network_imag = jax.grad(log_network_imag)
        grad_network = lambda params, data: jax.tree_map(lambda r, i: r + 1j * i, grad_network_real(params, data), grad_network_imag(params, data))

    def raveled_log_psi_grad(params, data):
        log_grads = grad_network(params, data)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def learning_rate_schedule(t_):
        return (1.0 / (1.0 + (t_ / args.lr0))) * jnp.sqrt(args.norm)

    def cal_update(params, prev_grad, data, e_l, t):
        param_flat, unravel_fn = jax.flatten_util.ravel_pytree(params)
        log_psi_grads = batch_raveled_log_psi_grad(params, data) / jnp.sqrt(args.batchsize)
        
        prev_grad_decayed = args.mu * prev_grad
        Ohat = log_psi_grads - jax.lax.pmean(jnp.mean(log_psi_grads, axis=0, keepdims=True), axis_name='batch')
        if args.dtype == 'complex':
            Ohat = jnp.concatenate([jnp.real(Ohat), jnp.imag(Ohat)], axis=0)

        if args.dtype != 'complex':
            median = jnp.median(jax.lax.all_gather(e_l, axis_name='batch'))
            tv = jax.lax.pmean(jnp.mean(jnp.abs(e_l - median)), axis_name='batch')
            diff = jnp.clip(e_l, median - args.clip_el * tv, median + args.clip_el * tv)
            diff = diff - jax.lax.pmean(jnp.mean(diff), axis_name='batch')
        else:
            e_l_real, e_l_imag = jnp.real(e_l), jnp.imag(e_l)
            median_real = jnp.median(jax.lax.all_gather(e_l_real, axis_name='batch'))
            median_imag = jnp.median(jax.lax.all_gather(e_l_imag, axis_name='batch'))
            tv_real = jax.lax.pmean(jnp.mean(jnp.abs(e_l_real - median_real)), axis_name='batch')
            tv_imag = jax.lax.pmean(jnp.mean(jnp.abs(e_l_imag - median_imag)), axis_name='batch')
            diff_real = jnp.clip(e_l_real, median_real - args.clip_el * tv_real, median_real + args.clip_el * tv_real)
            diff_imag = jnp.clip(e_l_imag, median_imag - args.clip_el * tv_imag, median_imag + args.clip_el * tv_imag)
            diff_real = diff_real - jax.lax.pmean(jnp.mean(diff_real), axis_name='batch')
            diff_imag = diff_imag - jax.lax.pmean(jnp.mean(diff_imag), axis_name='batch')
            diff = jnp.concatenate([diff_real, diff_imag], axis=0)


        np = Ohat.shape[-1]
        prev_grad_decayed, Ohat, diff = prev_grad_decayed.astype(jnp.float64), Ohat.astype(jnp.float64), diff.astype(jnp.float64)

        diff = -diff / jnp.sqrt(args.batchsize)
        epsilon_tilde = diff - Ohat @ prev_grad_decayed

        npad = (num_devices - Ohat.shape[-1] % num_devices) % num_devices
        Ohat = jnp.pad(Ohat, ((0, 0), (0, npad)), mode="constant")
        Ohat = jax.lax.all_to_all(Ohat.astype(jnp.float32), 'batch', 1, 0, tiled=True).astype(jnp.float64)

        T = jax.lax.psum(Ohat @ Ohat.T, axis_name='batch')
        B = T.shape[0]
        if args.dtype != 'complex':
            ones = jnp.ones((B, 1))
            reg = jnp.linalg.norm(T) * ones @ ones.T / B
        else:
            N = 2 * num_devices
            block_size = B // N
            rows = jnp.arange(N)[:, None]
            cols = jnp.arange(N)[None, :]
            base_pattern = ((rows + cols) % 2 == 0).astype(jnp.float32)
            reg = jnp.kron(base_pattern, jnp.ones((block_size, block_size))) * jnp.linalg.norm(T) / (B / 2)

        T_reg = T + reg + args.damping * jnp.eye(B)

        epsilon_tilde = jax.lax.all_gather(epsilon_tilde, axis_name='batch', axis=0, tiled=True)
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(T_reg, epsilon_tilde, assume_a="pos")
        dtheta_residual = jax.lax.all_gather(dtheta_residual, axis_name='batch', axis=0, tiled=True)[:np]

        grad = dtheta_residual + prev_grad_decayed
        grad_after_norm = grad * learning_rate_schedule(t) / jnp.linalg.norm(grad)
        if args.precision != 'x64':
            grad_after_norm = grad_after_norm.astype(jnp.float32)
        return unravel_fn(param_flat + grad_after_norm), grad, (jnp.linalg.norm(grad),)

    def spring_update_fn(params, prev_grad, data, t, subkeys):
        data, aux_mcmc = batch_mcmc_step(params, data, subkeys)
        e_l, aux_local = local_energy(params, data, {'cache': aux_mcmc['cache'], 't': t})
        if args.num_states > 1:
            e_l = jnp.trace(e_l, axis1=1, axis2=2)

        params, opt_state, aux = cal_update(params, prev_grad, data, e_l, t)
        logdict = {'loss': jnp.mean(e_l), 'pmove': jnp.mean(aux_mcmc['pmove']), 'variance': jnp.var(jax.lax.all_gather(e_l, axis_name='batch')),\
                    'lr': learning_rate_schedule(t) / aux[0], 'norm': aux[0]}
        aux_local = jax.tree_map(lambda x: jax.lax.pmax(x, axis_name='batch'), aux_local)
        logdict = logdict | aux_local
        logdict_new = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch').astype(jnp.float32), logdict)
        if args.dtype == 'complex':
            logdict_new['loss'] = jax.lax.pmean(logdict['loss'], axis_name='batch').astype(jnp.complex64)
        return params, opt_state, data, logdict_new

    return spring_update_fn
