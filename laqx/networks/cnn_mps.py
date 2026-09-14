"""CNN-MPS with shared state embeddings and positional encoding."""

from functools import partial

import jax
import jax.numpy as jnp

from laqx.networks import network_block


def _active_sample(pos, args):
    size = args.L1 * args.L2 * (1 if args.polarized else 2)
    return pos.reshape(-1)[:size]


def _symmetry_sample(pos, args):
    pos = _active_sample(pos, args)
    if args.polarized:
        pos = jnp.concatenate([pos, jnp.zeros_like(pos)])
    return pos.reshape(2, args.L1, args.L2)


def _network_dtype(args):
    return jnp.float64 if args.precision == 'x64' else jnp.float32


def init_cnn_mps(key, args):
    params = {}
    lattice = args.L1 * args.L2
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2

    key, input_key, pe_key = jax.random.split(key, 3)
    params['input'] = (
        jax.random.normal(input_key, shape=(physical, args.hidden)) * 0.01
    )
    params['pe'] = (
        jax.random.normal(pe_key, shape=(lattice, args.hidden)) * 0.01
    )

    params['cnn'] = []
    for _ in range(args.layers):
        key, subkey = jax.random.split(key)
        cnn = network_block.init_convolution(
            subkey,
            args.hidden,
            args.hidden,
            args.cutoff,
        )
        cnn['bias'] = jnp.zeros_like(cnn['bias'])
        params['cnn'].append({
            'norm': {'g': jnp.array(1.0), 'b': jnp.array(1.0)},
            'cnn': cnn,
        })

    key, subkey = jax.random.split(key)
    if args.MLP_layers == 0:
        limit = jnp.sqrt(6.0 / args.hidden)
        params['MLP'] = [{
            'w': jax.random.uniform(
                subkey,
                shape=(args.hidden, args.MLP_hidden),
                minval=-limit,
                maxval=limit,
            ),
            'b': jnp.zeros((args.MLP_hidden,)),
        }]
    else:
        params['MLP'] = network_block.init_MLP(
            subkey,
            args.hidden,
            args.MLP_hidden,
            args.MLP_layers,
        )

    output_dim = physical * args.mpsdim ** 2 * args.mps_num_head
    key, subkey = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (args.MLP_hidden + output_dim))
    output_w = jax.random.uniform(
        subkey,
        shape=(args.MLP_hidden, output_dim),
        minval=-limit,
        maxval=limit,
    )
    output_w /= (
        4 * jnp.linalg.norm(output_w) / jnp.sqrt(args.MLP_hidden)
    )
    params['output'] = {'w': output_w}
    return params


def _marshall_sign(pos, args):
    if not args.marshall:
        return jnp.array(1)

    up_spins = pos.reshape(args.L1, args.L2)
    sublattice = (
        jnp.arange(args.L1)[:, None] + jnp.arange(args.L2)[None, :]
    ) % 2
    parity = jnp.sum(up_spins * sublattice) % 2
    return 1 - 2 * parity


def apply_cnn_mps(params, pos, cache, args):
    del cache
    activation = network_block.get_activation(args.activation)
    spin_channels = network_block.get_spin_channels(args)
    physical = spin_channels * 2
    pos = _active_sample(pos, args)

    if args.polarized:
        local_state = pos
    else:
        indices = network_block.position_to_embedding_indices(
            pos,
            args,
            physical,
        )
        local_state = indices % physical
    hidden = params['input'][local_state] + params['pe']
    hidden = hidden.reshape(args.L1, args.L2, args.hidden)

    for index, layer in enumerate(params['cnn']):
        hidden = hidden + activation(
            network_block.convolution(
                hidden,
                args.boundary1,
                args.boundary2,
                **layer['cnn'],
                cutoff=args.cutoff,
            )
        )
        if index == args.layers - 1:
            hidden = hidden.astype(jnp.float64)
        hidden = network_block.layernorm(
            hidden,
            **layer['norm'],
            eps=1e-5,
        )

    hidden = hidden.reshape(args.L1 * args.L2, args.hidden)
    hidden = hidden.astype(_network_dtype(args))
    if args.MLP_layers == 0:
        hidden = network_block.linear_layer(hidden, **params['MLP'][0])
    else:
        hidden = network_block.apply_MLP(
            hidden,
            params['MLP'],
            activation,
        )

    mps = network_block.linear_layer(hidden, **params['output'])
    mps = mps.reshape(
        args.L1 * args.L2,
        physical,
        args.mpsdim,
        args.mpsdim,
        args.mps_num_head,
    )

    up_spins = (
        pos
        if args.polarized
        else pos.reshape(2, args.L1 * args.L2)[0]
    )
    mps_x = mps[jnp.arange(up_spins.shape[0]), up_spins]
    if args.boundary1 == 'obc':
        sign, logdet = network_block.multi_mps_contraction_obc(mps_x)
    else:
        sign, logdet = network_block.multi_mps_contraction_pbc(mps_x)
    sign *= _marshall_sign(up_spins, args).astype(sign.dtype)
    return {'sign': sign, 'logdet': logdet}


def make_cnn_mps(args):
    if args.dtype == 'complex':
        raise ValueError('cnn_mps currently supports only real parameters')

    init = partial(init_cnn_mps, args=args)
    apply = partial(apply_cnn_mps, cache=None, args=args)

    def symmetrized_apply(params, data, cache):
        del cache
        if not args.symmetry:
            return apply(params, data)

        data, weight = network_block.get_symmetry(
            [_symmetry_sample(data, args)],
            [1],
            args,
        )
        batched_data = jnp.stack(data, axis=0).reshape(
            -1,
            2 * args.L1 * args.L2,
        )
        weight = jnp.asarray(weight, dtype=_network_dtype(args))
        if args.use_boson:
            weight = jnp.ones_like(weight)

        new_cache = jax.vmap(apply, in_axes=(None, 0))(
            params,
            batched_data,
        )
        sign, logdet = network_block.combine_signed_logdet(
            new_cache['sign'],
            new_cache['logdet'],
            args,
            weight=weight,
        )
        return {'sign': sign, 'logdet': logdet}

    return init, symmetrized_apply, None
