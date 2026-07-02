# Developer Guide

This guide is for contributors who want to read or modify the LaQX code. It explains how a command travels through the codebase, what contracts connect the main modules, and where to start for common changes.

## Repository map

```text
.
├── main.py                 # CLI parser, environment setup, mode dispatch
├── laqx/
│   ├── train.py            # ground-state VMC: --mode march/spring/adam
│   ├── test.py             # checkpoint evaluation: --mode test
│   ├── tvmc.py             # time-dependent VMC: --mode tvmc
│   ├── gfmc.py             # Green's Function Monte Carlo: --mode gfmc
│   ├── pretrain.py         # restore-to-new-ansatz conversion used by train.py
│   ├── networks/           # wavefunction ansatz factories
│   ├── operators/          # Hamiltonians, observables, MCMC, local-energy application
│   └── utils/              # optimizers, TDVP, checkpointing, distributed helpers
├── docs/                   # MkDocs source, examples, and figures
├── mkdocs.yml              # documentation navigation
└── constraints.txt         # installation constraints
```

## Entrypoint and mode dispatch

All CLI options are defined in `main.py`. There are no argparse subcommands; the file parses one flat namespace, normalizes a few fields, and dispatches by `--mode`:

| `--mode` | Called function | What it does |
| --- | --- | --- |
| `march` | `laqx.train.train(args)` | VMC training with the MARCH optimizer. |
| `spring` | `laqx.train.train(args)` | VMC training with the SPRING optimizer. |
| `adam` | `laqx.train.train(args)` | VMC training with Adam. |
| `test` | `laqx.test.train(args)` | Load a checkpoint and measure an observable. |
| `gfmc` | `laqx.gfmc.train(args)` | Run GFMC projection from a checkpoint. |
| `tvmc` | `laqx.tvmc.train(args)` | Run TDVP/tVMC from a checkpoint or restore directory. |

## The main runtime pipeline

Most modes build the same three objects:

```text
args
  -> networks.network_provider(args)
       returns network_init, network_apply, orbitals
  -> operators.operator_provider(network_apply, args, inference=...)
       returns local_energy_or_observable(params, data, aux)
  -> operators.mcmc_provider(network_apply, args)
       returns a batched Metropolis update
```

The run-mode file then wraps these functions with JAX sharding, runs MCMC, evaluates local energies or observables, updates parameters when appropriate, and writes outputs.

## Run-mode files

### `laqx/train.py`

This file handles `march`, `spring`, and `adam`.

The main steps are:

1. Build the selected network with `network_provider(args)`.
2. Initialize walker configurations with `init_electrons()`.
3. Build the selected optimizer in `get_optimizer(args)`:
   - `utils/march.py` for `--mode march`,
   - `utils/spring.py` for `--mode spring`,
   - `utils/adam.py` for `--mode adam`.
4. Resume from the latest `ckpt_*` in `args.output` if present.
5. If `args.output` has no checkpoint and `--restore` is set, call `pretrain.py` to convert the restored checkpoint into a `ckpt_000000.npz` compatible with the new run.
6. Optionally run MCMC-only burn-in when `--burn_in` is set.
7. Run the training loop, write `log.csv`, and save checkpoints every `--save_frequency` steps.

Training checkpoints are pickled dictionaries with `t`, `data`, `params`, and `opt_state`.

### `laqx/test.py`

This file handles `--mode test`.

It expects `args.output` to contain a checkpoint. It loads the latest checkpoint, runs `drop_step` MCMC steps, then runs `steps` measurement iterations. Output depends on the observable:

- single-state `energy` writes `result.txt`,
- single-state `polar` writes `polar.txt`,
- matrix or grouped observables write pickle payloads such as `energy.npz`, `fermi2.npz`, or `localo.npz`.

### `laqx/tvmc.py`

This file handles `--mode tvmc`.

It can create a fresh tVMC output directory from `--restore`. In that case it copies data and parameters from the restored checkpoint, and for PEFT networks such as `ace_peft`, it initializes a split parameter tree with `fixed` and `peft` subtrees.

The TDVP update is implemented in `laqx/utils/tdvp.py`. The update only modifies the `peft` subtree when the parameter tree is split.

When `--obs density` is used, tVMC accumulates density measurements and writes `density.npz` at the end.

### `laqx/gfmc.py`

This file handles `--mode gfmc`.

It loads a checkpoint from `args.output`, runs burn/drop MCMC steps, then performs GFMC propagation using `operators/gfmc_utils.py`. It writes `gfmc.csv` and does not save new checkpoints.

### `laqx/pretrain.py`

This is not a public `--mode` target. `train.py` calls it when `--restore` is used to initialize a fresh output directory. It writes a pretrain-style checkpoint with `t = -1`.

## Network contract

Network factories live in `laqx/networks/` and are registered in `laqx/networks/__init__.py`. A factory returns:

```python
network_init, network_apply, orbitals = make_network(args)
```

The pieces mean:

- `network_init(key, ...)` creates the parameter PyTree.
- `network_apply(params, data, cache)` evaluates the wavefunction on one configuration.
- `orbitals` is optional; some workflows use it for orbital-level access.

Single-state networks return a cache dictionary containing:

```python
{
    "sign": sign,
    "logdet": log_abs_or_complex_amplitude,
    ...
}
```

The local-energy and MCMC code use these fields to form amplitude ratios.

NES networks return:

```python
{
    "psi": state_amplitudes,
}
```

where `psi` has one amplitude per optimized state.

Fast-update networks may also store determinant inverses, local orbitals, or a `neighbor` value in the cache. The currently important fast-update implementations are `tensor` and `scale`; other networks generally recompute amplitudes directly.

When adding a new ansatz:

1. Add `laqx/networks/my_network.py`.
2. Implement `make_my_network(args)` returning `(init, apply, orbitals)`.
3. Make `apply` return the correct cache convention (`sign`/`logdet` for single-state, `psi` for NES).
4. Register the network in `laqx/networks/__init__.py`.
5. Add any new CLI parameters to `main.py` only if existing fields are insufficient.

## Operator contract

Hamiltonians and observables live under `laqx/operators/hamiltonian/` and are registered in `laqx/operators/__init__.py`. The full `t`/`a`/`U`/`h` operator dictionary contract is documented in [Custom Hamiltonians](custom-hamiltonian.md).
