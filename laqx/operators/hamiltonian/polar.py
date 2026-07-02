import jax
import jax.numpy as jnp
import numpy as np

def make_polar(args):
    def _c(params, data, cache):
        data = data.reshape(-1, 2, args.L1, args.L2)
        charge = jnp.sum(data, axis=(1, 2)) * jnp.arange(args.L2) * (2 * np.pi / args.L2)
        polar = jnp.exp(jnp.sum(charge, axis=-1) * 1j)
        logdict = {}
        if args.fast_update:
            logdict['neighbor'] = jnp.max(cache['cache']['neighbor'])
        return polar, logdict
    return _c

