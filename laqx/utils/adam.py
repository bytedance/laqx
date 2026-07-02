import jax
import jax.numpy as jnp
import optax

def get_adam_update_fn(network, local_energy, args, batch_mcmc_step, num_devices):
    def learning_rate_schedule(t_):
        return args.lr * (1.0 / (1.0 + (t_ / args.lr0)))
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(learning_rate_schedule), optax.scale(-1))

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

    batch_grad = jax.vmap(grad_network, in_axes=(None, 0), out_axes=-1)

    def update_fn(params, opt_state, data, t, subkeys):
        data, aux_mcmc = batch_mcmc_step(params, data, subkeys)
        e_l, aux_local = local_energy(params, data, aux_mcmc)
        if args.num_states > 1:
            e_l = jnp.trace(e_l, axis1=1, axis2=2)

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
            diff = diff_real + 1j * diff_imag
            diff = diff - jax.lax.pmean(jnp.mean(diff), axis_name='batch')
            diff = diff.conj()

        grad = batch_grad(params, data)
        grad = jax.tree_map(lambda x: jnp.real(jax.lax.pmean(jnp.mean(x * diff, axis=-1), axis_name='batch')), grad)
        if args.precision != 'x64':
            grad = jax.tree_map(lambda x: x.astype(jnp.float32), grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        logdict = {'loss': jnp.mean(e_l), 'pmove': jnp.mean(aux_mcmc['pmove']), 'variance': jnp.var(jax.lax.all_gather(e_l, axis_name='batch')),\
                    'lr': learning_rate_schedule(t)}
        aux_local = jax.tree_map(lambda x: jax.lax.pmax(x, axis_name='batch'), aux_local)
        logdict = logdict | aux_local
        logdict_new = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch').astype(jnp.float32), logdict)
        if args.dtype == 'complex':
            logdict_new['loss'] = jax.lax.pmean(logdict['loss'], axis_name='batch').astype(jnp.complex64)
        return params, opt_state, data, logdict_new

    return update_fn, optimizer
