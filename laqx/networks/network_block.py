import jax
import jax.numpy as jnp
from jax.nn import softmax
import numpy as np

def get_activation(name):
  if name == "silu":
    return jax.nn.silu
  if name == "tanh":
    return jnp.tanh
  if name == "poly":
    return lambda x: x - x**3 / 3
  raise NotImplementedError

def init_linear_layer(key, in_dim, out_dim, include_bias: bool = True, use_zeros: bool = False):
  key1, key2 = jax.random.split(key)
  weight = jax.random.normal(key1, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim))
  if use_zeros:
      weight = jnp.zeros((in_dim, out_dim))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}
  
def linear_layer(x, w, b = None):
  y = jnp.dot(x, w)
  return y + b if b is not None else y


def init_mps(key, in_dim, out_dim):
  return init_linear_layer(key, in_dim, out_dim)


def init_layernorm(feature_dim):
  gamma = jnp.ones((feature_dim,))
  beta = jnp.zeros((feature_dim,))
  return {'g': gamma, 'b': beta}

def layernorm(x, g, b, eps = 1e-4):
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  
  x_normalized = (x - mean) / jnp.sqrt(var + eps)
  return x_normalized * g + b



def init_MLP(key, in_dim, hidden_dim, layers, last=False):
    MLP = []
    key, subkey = jax.random.split(key)
    MLP.append(init_linear_layer(subkey, in_dim, hidden_dim))
    for _ in range(layers-1):
        key, subkey = jax.random.split(key)
        MLP.append(init_linear_layer(subkey, hidden_dim, hidden_dim))
    return MLP

def apply_MLP(hidden, MLP, activation):
    for layer in MLP:
      hidden = activation(linear_layer(hidden, **layer))
    return hidden

def scaled_dot_product(q, k, v):
  d_k = q.shape[-1]
  attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1), precision='highest')
  attn_logits = attn_logits / jnp.sqrt(d_k)
  attention = softmax(attn_logits, axis=-1)
  values = jnp.matmul(attention, v)
  return values

def init_convolution(key, dim_in, dim_out, cutoff=3, use_bias=True):
  kernel_key, bias_key = jax.random.split(key)
  kernel = jax.random.normal(kernel_key, (cutoff, cutoff, dim_in, dim_out)) / jnp.sqrt(float(cutoff ** 2 * dim_in))
  if use_bias:
    bias = jax.random.normal(bias_key, (dim_out,))
    return {'kernel': kernel, 'bias': bias}
  else:
    return {'kernel': kernel}

def circular_pad(x, boundary1, boundary2, size=1):
  mode1 = 'constant' if boundary1 == 'obc' else 'wrap'
  mode2 = 'constant' if boundary2 == 'obc' else 'wrap'

  x = jnp.pad(x, ((size, size), (0, 0), (0, 0)), mode=mode1)
  x = jnp.pad(x, ((0, 0), (size, size), (0, 0)), mode=mode2)
  return x

def convolution(x, boundary1, boundary2, kernel, bias=None, cutoff=3):
  x_padded = circular_pad(x, boundary1, boundary2, size=cutoff // 2)
  y = jax.lax.conv_general_dilated(x_padded[None], kernel, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
  return y + bias if bias is not None else y

def nearby_convolution(site, x, args, kernel, bias):
  x_padded = circular_pad(x, args.boundary1, args.boundary2, size=args.cutoff - 1)
  i, j = site // args.L2, site % args.L2
  x_env = jax.lax.dynamic_slice(x_padded, (i, j, 0), (2 * args.cutoff - 1, 2 * args.cutoff - 1, x.shape[-1]))
  y = jax.lax.conv_general_dilated(x_env[None], kernel, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
  y = y + bias
  return y


def nearby_convolution_tprime(site, x, args, kernel, bias):
  move = args.cutoff // 2
  x_padded = circular_pad(x, args.boundary1, args.boundary2, size=args.cutoff)
  i_coords = site // args.L2
  j_coords = site % args.L2
  min_i = jnp.min(i_coords)
  min_j = jnp.min(j_coords)
  max_i = jnp.max(i_coords)
  max_j = jnp.max(j_coords)
  min_i = jnp.where(max_i - min_i <= 1, min_i, max_i)
  min_j = jnp.where(max_j - min_j <= 1, min_j, max_j)

  x_env = jax.lax.dynamic_slice(x_padded, (min_i+1, min_j+1, 0), (2 * args.cutoff, 2 * args.cutoff, x.shape[-1]))
  y = jax.lax.conv_general_dilated(x_env[None], kernel, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
  y = y + bias
  return y

def init_multihead_attention(key, dim):
  key, subkey = jax.random.split(key)
  qkv_proj = init_linear_layer(subkey, dim, 3 * dim, include_bias=False)
  key, subkey = jax.random.split(key)
  out_proj = init_linear_layer(subkey, dim, dim, include_bias=False)
  return {'qkv': qkv_proj, 'out': out_proj}

def multihead_attention(x, num_head, qkv, out):
  qkv = linear_layer(x, **qkv)
  qkv = qkv.reshape(x.shape[0], num_head, -1)
  # q, k, v = jnp.split(qkv, 3, axis=-1)
  # value = triton_util.triton_fa_mixed_online(q, k, v)
  qkv = qkv.transpose(1, 0, 2)
  q, k, v = jnp.split(qkv, 3, axis=-1)
  value = scaled_dot_product(q, k, v).transpose(1, 0, 2)
  value = value.reshape(x.shape)
  return linear_layer(value, **out)

def compute_inv_det_lu(mat, V=None):
    N = mat.shape[0]
    lu, piv = jax.scipy.linalg.lu_factor(mat)

    diag_u = jnp.diag(lu)
    
    num_swaps = jnp.sum(jnp.arange(N) != piv)
    sign = jnp.power(-1.0, num_swaps) * jnp.prod(jnp.sign(diag_u))
    
    log_det = jnp.sum(jnp.log(jnp.abs(diag_u)))
    
    b = jnp.eye(N, dtype=mat.dtype) if V is None else V
    inv = jax.scipy.linalg.lu_solve((lu, piv), b)
    
    return inv, (sign, log_det)


def find_changed_columns(matrix_old, matrix_new, size, tol=1e-6):
    col_diff_magnitudes = jnp.mean(jnp.abs(matrix_new - matrix_old), axis=(0, 2)) / jnp.mean(jnp.abs(matrix_old), axis=(0, 2))
    changed_indices = jnp.where(col_diff_magnitudes > tol, size=size, fill_value=-1)[0]
    changed_total = jnp.sum(jnp.where(col_diff_magnitudes > tol, 1, 0))
    return changed_indices, changed_total


def fast_update(Phi, Phi_prime, cache, changed_indices):
    V = (Phi_prime[changed_indices] - Phi[changed_indices])
    inv, sign, logdet = cache
    inv_U = inv[:, changed_indices] * jnp.where(changed_indices == -1, 0, 1)
    A = jnp.matmul(V, inv_U)
    A = A + jnp.eye(A.shape[0], dtype=A.dtype)

    inv_A, (sign_A, logdet_A) = compute_inv_det_lu(A, V)
    sign = sign * sign_A
    logdet = logdet + logdet_A
    inv = inv - jnp.matmul(inv_U, jnp.matmul(inv_A, inv))
    return (inv, sign, logdet)



def get_nearby(site, args):
  move = args.cutoff // 2
  offsets = jnp.arange(-move, move + 1)
  delta_i = jnp.repeat(offsets, args.cutoff)
  delta_j = jnp.tile(offsets, args.cutoff)
  site_2d = site[:, None]
  i_coords = site_2d // args.L2
  j_coords = site_2d % args.L2
  neighbor_i = i_coords + delta_i
  neighbor_j = j_coords + delta_j
  if args.boundary1 != 'obc':
    neighbor_i = neighbor_i % args.L1
  if args.boundary2 != 'obc':
    neighbor_j = neighbor_j % args.L2
  nearby_sites = neighbor_i * args.L2 + neighbor_j
  is_valid = (neighbor_i >= 0) & (neighbor_i < args.L1) & (neighbor_j >= 0) & (neighbor_j < args.L2)
  nearby_sites = jnp.where(is_valid, nearby_sites, site_2d)
  nearby_sites = nearby_sites.reshape(-1)
  return nearby_sites, is_valid

def get_nearby_tprime(site, args):
  i_coords = site // args.L2
  j_coords = site % args.L2
  min_i = jnp.min(i_coords)
  min_j = jnp.min(j_coords)
  max_i = jnp.max(i_coords)
  max_j = jnp.max(j_coords)
  min_i = jnp.where(max_i - min_i <= 1, min_i, max_i)
  min_j = jnp.where(max_j - min_j <= 1, min_j, max_j)

  move = args.cutoff // 2
  offsets = jnp.arange(-move, move + 2)
  delta_i = jnp.repeat(offsets, args.cutoff + 1)
  delta_j = jnp.tile(offsets, args.cutoff + 1)
  neighbor_i = min_i + delta_i
  neighbor_j = min_j + delta_j
  if args.boundary1 != 'obc':
    neighbor_i = neighbor_i % args.L1
  if args.boundary2 != 'obc':
    neighbor_j = neighbor_j % args.L2
  nearby_sites = neighbor_i * args.L2 + neighbor_j
  is_valid = (neighbor_i >= 0) & (neighbor_i < args.L1) & (neighbor_j >= 0) & (neighbor_j < args.L2)
  nearby_sites = jnp.where(is_valid, nearby_sites, min_i * args.L2 + min_j)
  return nearby_sites, is_valid


def get_nearby_tprime_scatter(site, args):
  i_coords = site // args.L2
  j_coords = site % args.L2
  min_i = jnp.min(i_coords)
  min_j = jnp.min(j_coords)
  max_i = jnp.max(i_coords)
  max_j = jnp.max(j_coords)
  min_i = jnp.where(max_i - min_i <= 1, min_i, max_i)
  min_j = jnp.where(max_j - min_j <= 1, min_j, max_j)

  move = args.cutoff // 2
  offsets = jnp.arange(-move, move + 2)
  delta_i = jnp.repeat(offsets, args.cutoff + 1)
  delta_j = jnp.tile(offsets, args.cutoff + 1)
  neighbor_i = min_i + delta_i
  neighbor_j = min_j + delta_j
  if args.boundary1 != 'obc':
    neighbor_i = neighbor_i % args.L1
  if args.boundary2 != 'obc':
    neighbor_j = neighbor_j % args.L2
  # nearby_sites = neighbor_i * args.L2 + neighbor_j
  # is_valid = (neighbor_i >= 0) & (neighbor_i < args.L1) & (neighbor_j >= 0) & (neighbor_j < args.L2)
  # nearby_sites = jnp.where(is_valid, nearby_sites, min_i * args.L2 + min_j)

  dist_i = jnp.abs(neighbor_i.reshape(36, 1) - i_coords.reshape(1, 2))
  dist_j = jnp.abs(neighbor_j.reshape(36, 1) - j_coords.reshape(1, 2))

  if (args.boundary1 != 'obc'):
    dist_i = jnp.minimum(dist_i, args.L1 - dist_i)

  if (args.boundary2 != 'obc'):
    dist_j = jnp.minimum(dist_j, args.L2 - dist_j)

  mask = jnp.where((jnp.min(dist_i ** 2 + dist_j ** 2, axis=-1) <= 4) & (neighbor_i >= 0) & (neighbor_i < args.L1) & (neighbor_j >= 0) & (neighbor_j < args.L2), 1, 0)
  reduce = jnp.nonzero(mask, size=18, fill_value=-1)[0]
  neighbor_i, neighbor_j = neighbor_i[reduce], neighbor_j[reduce]
  nearby_sites = neighbor_i * args.L2 + neighbor_j
  is_valid = jnp.where(reduce != -1, 1, 0)
  nearby_sites = jnp.where(is_valid, nearby_sites, nearby_sites[0])
  return nearby_sites, is_valid, reduce




def get_symmetry(data, weight, args):
  use_fermion_ph = args.model != 'spin' and (args.particle_hole or ('p' in args.symmetry and (2 * args.particles_up != args.particles)))
  if use_fermion_ph:
     data, weight = get_ph(data, weight, args)
     
  if '2' in args.symmetry or '4' in args.symmetry:
      data = [data[0], jnp.rot90(data[0], k=2, axes=(-2, -1))]
      Nud = jnp.sum(data[0], axis=(-1, -2))
      flip = jnp.sum(Nud * (Nud - 1) // 2)
      weight = [weight[0], weight[0] * ((-1) ** flip)]

  if '4' in args.symmetry:
      def get_c4_sign_matrix(L1, L2):
        indices = jnp.arange(L1 * L2).reshape(L1, L2)
        perm = jnp.rot90(indices, k=1, axes=(-2, -1)).flatten()
      
        dim = L1 * L2
        r = jnp.arange(dim).reshape(-1, 1)
        c = jnp.arange(dim).reshape(1, -1)
        
        K = ((r < c) & (perm[r] > perm[c])).astype(jnp.int32)
        return K
      
      K_matrix = get_c4_sign_matrix(args.L1, args.L2)
      all_symmetries = []
      all_weight = []
      for i in range(len(data)):
          all_symmetries.append(data[i])
          all_weight.append(weight[i])
          
          rotated_data = jnp.rot90(data[i], k=1, axes=(-2, -1))
          all_symmetries.append(rotated_data)
        
          flat_config = rotated_data.reshape(2, -1)
          swaps = jnp.einsum('sa,ab,sb->', flat_config, K_matrix, flat_config)
          sign = (-1) ** swaps
          all_weight.append(weight[i] * sign * args.rotation)
          
      data = all_symmetries
      weight = all_weight

  if 'D' in args.symmetry:
      all_symmetries = []
      all_weight = []
      for i in range(len(data)):
          # if args.hp_type != 'obc':
          all_symmetries.append(data[i])
          all_symmetries.append(jnp.flip(data[i], axis=1))
          all_weight.append(weight[i])
          A = jnp.sum(data[i], axis=-1)
          AA = A.reshape(2, args.L1, 1) * A.reshape(2, 1, args.L1)
          element = (jnp.sum(AA) - jnp.sum(jnp.trace(AA, axis1=1, axis2=2))) // 2
          mul = 1 if args.particle_hole else 1
          all_weight.append(weight[i] * ((-1) ** element) * mul)
          # else:
          # all_symmetries.append(data[i])
          # all_symmetries.append(jnp.flip(data[i], axis=2))
          # all_weight.append(weight[i])
          # A = jnp.sum(data[i], axis=-1)
          # element = jnp.sum(A * (A - 1) // 2)
          # mul = 1 if args.particle_hole else 1
          # all_weight.append(weight[i] * ((-1) ** element) * mul)

      data = all_symmetries
      weight = all_weight
  
  if 'p' in args.symmetry:
      all_symmetries = []
      all_weight = []
      for i in range(len(data)):
          all_symmetries.append(data[i])
          all_weight.append(weight[i])

          if args.model == 'spin':
            all_symmetries.append(data[i].at[0].set(1 - data[i][0]))
            all_weight.append(weight[i])
          else:
            all_symmetries.append(jnp.flip(data[i], axis=0))
            N = jnp.sum(data[i], axis=(-1, -2))
            if args.particle_hole:
              all_weight.append(weight[i] * ((-1) ** (N[0] * (N[1] + 1))))
            else:
              all_weight.append(weight[i] * ((-1) ** jnp.prod(N)))
      data = all_symmetries
      weight = all_weight

  def get_sign_X(data, p):
      A = jnp.sum(data[:, :args.L1 - p], axis=(-1, -2))
      B = jnp.sum(data[:, args.L1 - p:], axis=(-1, -2))
      return (-1) ** (jnp.sum(A * B))

  Lx = args.L1
  Ly = args.L2
  translate_x = np.exp(1j * 2 * np.pi * args.kx / Lx)
  translate_y = np.exp(1j * 2 * np.pi * args.ky / Ly)
  if args.dtype != 'complex':
     translate_x = np.real(translate_x)
     translate_y = np.real(translate_y)
  def apply_shift_X(data, weight, shift):
      all_symmetries = []
      all_weight = []
      for i in range(len(data)):
          for p in range(0, args.L1, shift):
              all_symmetries.append(jnp.roll(data[i], p, axis=1))
              mul = translate_x ** p
              all_weight.append(weight[i] * get_sign_X(data[i], p) * mul)
      return all_symmetries, all_weight

  if args.boundary1 == 'pbc':
    if 'X' in args.symmetry:
        data, weight = apply_shift_X(data, weight, args.L1 // 2)

    elif 'S' in args.symmetry:
        data, weight = apply_shift_X(data, weight, args.L1 // 4)

    elif 'T' in args.symmetry:
        if args.model == 'hofstadter':
          T = int(1 / args.alpha)
        elif args.model == 'haldane':
          T = 2
        else:
          T = 1
        data, weight = apply_shift_X(data, weight, T)

  def get_sign_Y(data, p):
      A = jnp.sum(data[:, :, :args.L2 - p], axis=-1)
      B = jnp.sum(data[:, :, args.L2 - p:], axis=-1)
      return (-1) ** jnp.sum(A * B)

  def apply_shift_Y(data, weight, shift):
      all_symmetries = []
      all_weight = []
      for i in range(len(data)):
          for p in range(0, args.L2, shift):
              all_symmetries.append(jnp.roll(data[i], p, axis=2))
              mul = translate_y ** p
              all_weight.append(weight[i] * get_sign_Y(data[i], p) * mul)
      return all_symmetries, all_weight

  if args.boundary2 == 'pbc':
    if 'Y' in args.symmetry:
        data, weight = apply_shift_Y(data, weight, args.L2 // 2)

    elif 'S' in args.symmetry:
        data, weight = apply_shift_Y(data, weight, args.L2 // 4)

    elif 'T' in args.symmetry:
        data, weight = apply_shift_Y(data, weight, 1)

  if use_fermion_ph:
     data, weight = get_ph(data, weight, args)

  return data, weight



def get_ph(data, weight, args):
  flip_vector = jnp.array([0, 1]).reshape(2, 1, 1)
  sign_vector = (jnp.arange(args.L1)[:, None] + jnp.arange(args.L2)).reshape(-1) % 2


  all_symmetries = []
  all_weight = []
  for i in range(len(data)):
      single_data = data[i].reshape(2, args.L1 * args.L2)[1]
      before = jnp.sum(jnp.cumsum(single_data) - single_data)
      all_symmetries.append(jnp.abs(flip_vector - data[i]))
      all_weight.append(weight[i] * ((-1) ** (before + jnp.sum(single_data * sign_vector))))

  data, weight = all_symmetries, all_weight
  return data, weight


def get_spin_channels(args):
  return 1 if args.polarized else 2


def get_occupied_indices(pos, args):
  return jnp.nonzero(pos, size=args.particles)[0]


def position_to_embedding_indices(pos, args, physical):
  lattice = args.L1 * args.L2
  if args.polarized:
    pos_up = pos.reshape(2, -1)[0]
    return pos_up + physical * jnp.arange(lattice)
  pos = pos.reshape(2, -1).transpose(1, 0)
  return jnp.dot(pos, 2 ** jnp.arange(2)) + physical * jnp.arange(lattice)


def combine_signed_logdet(sign, logdet, args, weight=None):
  max_logdet = jax.lax.stop_gradient(jnp.max(logdet))
  det = jnp.exp(logdet - max_logdet)
  mask = jnp.abs(det) > 0.0
  factor = 1 if weight is None else weight
  result = jnp.sum(sign * det * factor, where=mask)
  if args.dtype == 'complex':
    logdet = jnp.log(result) + max_logdet
    sign = jnp.ones_like(logdet, dtype=jnp.float32)
  else:
    sign = jnp.sign(result)
    logdet = jnp.log(jnp.abs(result)) + max_logdet
  return sign, logdet


def symmetrize_apply(params, data, cache, args, apply_fn, apply_fast_fn=None):
  if not args.symmetry:
    if cache and args.fast_update and apply_fast_fn is not None:
      return apply_fast_fn(params, data, cache)
    return apply_fn(params, data)

  data = data.reshape(2, args.L1, args.L2)
  data, weight = get_symmetry([data], [1], args)
  batched_data = jnp.stack(data, axis=0).reshape(-1, 2 * args.L1 * args.L2)
  weight = jnp.array(weight)

  if cache and args.fast_update and apply_fast_fn is not None and 'in_cache' in cache:
    in_cache = cache['in_cache']
    new_cache = jax.vmap(apply_fast_fn, in_axes=(None, 0, 0))(params, batched_data, in_cache)
  else:
    new_cache = jax.vmap(apply_fn, in_axes=(None, 0))(params, batched_data)

  out_cache = {'in_cache': new_cache}
  if 'neighbor' in new_cache:
    out_cache['neighbor'] = jnp.max(new_cache['neighbor'])

  sign, logdet = combine_signed_logdet(new_cache['sign'], new_cache['logdet'], args, weight=weight)
  out_cache['sign'], out_cache['logdet'] = sign, logdet
  return out_cache

def multi_mps_contraction_pbc(mps_x):
    _, mpsdim, _, num_head = mps_x.shape
    eye = jnp.eye(mpsdim, dtype=mps_x.dtype)
    state = jnp.repeat(eye[:, :, None], num_head, axis=-1)

    def scan_fn(state, mps_i):
        next_state = jnp.einsum('abn,bcn->acn', state, mps_i)
        scale = jnp.max(jnp.abs(next_state))
        return next_state / scale, jnp.log(scale)

    final_state, renorm = jax.lax.scan(scan_fn, state, mps_x)
    contracted = jnp.sum(jnp.trace(final_state, axis1=0, axis2=1))
    log_scale = jnp.sum(renorm)
    return jnp.sign(contracted), jnp.log(jnp.abs(contracted)) + log_scale
