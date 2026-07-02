import jax
import jax.numpy as jnp

from functools import partial
from laqx.operators import ham_utils

def switch(operator, x1, key):
  t_operator, t_weights = operator['t']
  a_operator, a_weights = operator['a']

  if t_operator is not None:
    def t_switch(key, array):
      check = ham_utils.check_first
      def check_prob(array, operator):
        return (jnp.maximum(jnp.abs(t_weights), 0.05) * jax.vmap(check, in_axes=(None, 0))(array, operator)).astype(jnp.float64)

      apply_prob = check_prob(array, t_operator)
      operator = jax.random.choice(key, t_operator, p=apply_prob)
      _, new_array = ham_utils.apply_first(array, operator)
      apply_sign_rev = check_prob(new_array, t_operator)
      return new_array, jnp.log(jnp.sum(apply_prob) / jnp.sum(apply_sign_rev))
  else:
    t_switch = None

  if a_operator is not None:
    def a_switch(key, array):
      check = ham_utils.check_second
      def check_prob(array, operator):
        return (jnp.maximum(jnp.abs(a_weights), 0.05) * jax.vmap(check, in_axes=(None, 0))(array, operator)).astype(jnp.float64)

      apply_prob = check_prob(array, a_operator)
      operator = jax.random.choice(key, a_operator, p=apply_prob)
      _, new_array = ham_utils.apply_second(array, operator)
      apply_sign_rev = check_prob(new_array, a_operator)
      return new_array, jnp.log(jnp.sum(apply_prob) / jnp.sum(apply_sign_rev))
  else:
    a_switch = None

  if t_switch is not None and a_switch is not None:
    key, subkey1, subkey2 = jax.random.split(key, 3)
    use_switch = jax.random.bernoulli(subkey1, p=0.5)
    x2_t, fix_t = t_switch(subkey2, x1)
    x2_a, fix_a = a_switch(subkey2, x1)
    x2 = jnp.where(use_switch[..., None], x2_t, x2_a)
    fix = jnp.where(use_switch, fix_t, fix_a)
  elif t_switch is not None:
    key, subkey = jax.random.split(key)
    x2, fix = t_switch(subkey, x1)
  elif a_switch is not None:
    key, subkey = jax.random.split(key)
    x2, fix = a_switch(subkey, x1)
  else:
    raise NotImplementedError
  return x2, fix, key



def mh_update(params, f, operator, x1, key, cache, num_accepts, args):
  x2, fix, key = switch(operator, x1, key)
  
  lp_1 = 2 * cache['logdet']
  new_cache = f(params, x2, cache)
  
  ratio = jnp.real(2 * new_cache['logdet'] - lp_1) + fix

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=lp_1.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  cache = jax.tree_map(lambda new, old: jnp.where(jnp.reshape(cond, cond.shape + (1,) * (new.ndim - cond.ndim)), new, old), new_cache, cache)

  num_accepts += jnp.sum(cond)
  return x_new, key, cache, num_accepts


def nes_mh_update(params, f, operator, x1, key, cache, num_accepts, args):
  key, subkey = jax.random.split(key)
  state_index = jax.random.choice(subkey, args.num_states)
  x2, fix, key = switch(operator, x1[state_index], key)
  
  x2_new = x1.at[state_index].set(x2)
  lp_1 = 2 * cache['logdet']

  x2_cache = f(params, x2, cache)
  new_cache = {'psi': cache['psi'].at[state_index].set(x2_cache['psi'])}
  sign, logdet = jnp.linalg.slogdet(new_cache['psi'])
  new_cache['sign'] = sign
  new_cache['logdet'] = logdet
  
  ratio = jnp.real(2 * new_cache['logdet'] - lp_1) + fix

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=lp_1.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2_new, x1)
  cache = jax.tree_map(lambda new, old: jnp.where(jnp.reshape(cond, cond.shape + (1,) * (new.ndim - cond.ndim)), new, old), new_cache, cache)

  num_accepts += jnp.sum(cond)
  return x_new, key, cache, num_accepts

def make_mcmc_step(network, operator, args):
  def mcmc_step(params, data, key):
    if args.num_states == 1:
      cache = network(params, data, None)
    else:
      cache = jax.vmap(lambda d: network(params, d, None))(data)
      sign, logdet = jnp.linalg.slogdet(cache['psi'])
      cache['sign'] = sign
      cache['logdet'] = logdet
    
    def step_fn(i, x):
      if args.num_states == 1:
        return mh_update(params, network, operator, *x, args)
      else:
        return nes_mh_update(params, network, operator, *x, args)

    data, key, cache, num_accepts = jax.lax.fori_loop(0, args.mcmc_step, step_fn, (data, key, cache, 0.))
    pmove = jnp.sum(num_accepts) / args.mcmc_step
    return data, {'pmove': pmove, 'cache': cache, 'key': key}

  return mcmc_step
