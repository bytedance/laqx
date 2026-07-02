# Workflow

This page explains the main ground-state VMC workflow implemented in `laqx/train.py`. It focuses on the path used by `--mode march`, `--mode spring`, and `--mode adam`, where LaQX samples Monte Carlo walkers, evaluates local energies, updates neural-network parameters, and writes logs and checkpoints.

## Where `train.py` fits

The command-line entry point is `main.py`. After parsing arguments, it sends the three ground-state training modes to `laqx.train.train(args)`:

```text
python main.py --mode march ...
python main.py --mode spring ...
python main.py --mode adam ...

main.py
  -> laqx.train.train(args)
```

Inside `train.py`, all three modes share the same high-level structure:

```text
select network
  -> initialize parameters and walkers
  -> build local-energy and MCMC functions from the Hamiltonian
  -> choose an optimizer update rule
  -> optionally resume from a checkpoint
  -> repeat: MCMC sample, evaluate local energy, update parameters, log, checkpoint
```

## 1. Build the neural wavefunction

Training starts by asking the network registry for the selected ansatz:

```python
network_init, network_apply, _ = networks.network_provider(args)
params = network_init(subkey)
```

`network_provider(args)` uses `--network_name` to return the factory registered in `laqx/networks/__init__.py`. The returned functions have a simple contract:

- `network_init(key)` creates the parameter PyTree.
- `network_apply(params, data, cache)` evaluates the wavefunction on a configuration.

The training loop keeps `network_apply` and passes it to the operator and optimizer builders. This keeps the optimizer code independent of the concrete ansatz implementation.

## 2. Initialize electron configurations

`init_electrons(key, args)` creates the initial Monte Carlo walkers. It builds one binary occupation vector for spin up and one for spin down, randomly permutes each vector, and concatenates them:

```text
[spin-up occupations, spin-down occupations]
```

For a spinful lattice with `L1 * L2` spatial sites, the output has `2 * L1 * L2` occupation entries per walker. `--particles_up` controls the number of spin-up particles, and `--particles - --particles_up` controls the number of spin-down particles.

When `--num_states > 1`, the walker batch is reshaped to include a state axis. This is the Neural Excited States (NES) path, where one Monte Carlo sample carries configurations for multiple optimized states.

## 3. Create sharded training data

The batch dimension is the parallel axis used by the compiled update function:

```python
pbatch = P("batch")
mesh = jax.make_mesh((total_devices,), ("batch",))
data = jax.device_put(data, jax.sharding.NamedSharding(mesh, pbatch))
```

In practice, this means each training step sees a sharded batch of walkers. The optimizer update function is later wrapped with `shard_map` and `jax.jit`, so the same step function can run efficiently across local accelerator devices.

## 4. Build local energy and MCMC functions

The nested `get_optimizer(args)` function constructs the two physics functions used by every optimizer:

```python
local_energy = operators.operator_provider(network_apply, args)
batch_mcmc_step = operators.mcmc_provider(network_apply, args)
```

These functions connect the ansatz to the Hamiltonian contract described in [Custom Hamiltonians](custom-hamiltonian.md):

- `operator_provider(...)` builds a local-energy function. It applies the registered Hamiltonian groups (`t`, `a`, `U`, `h`) to compute $E_\mathrm{loc}(x)$ for sampled configurations.
- `mcmc_provider(...)` builds a batched Metropolis update. Its proposals come from the off-diagonal Hamiltonian terms, and its acceptance ratio is computed with `network_apply`.

This separation is useful when extending LaQX: new Hamiltonians only need to satisfy the operator contract, and new ansatzes only need to satisfy the network contract. The training loop does not need to know their internal details.

## 5. Choose the optimizer update

The `--mode` option selects the update rule:

| Mode | Builder | Optimizer state | Main idea |
| --- | --- | --- | --- |
| `march` | `march.get_march_update_fn(...)` | previous gradient and adaptive scale vector | Natural-gradient-style update with MARCH preconditioning. |
| `spring` | `spring.get_spring_update_fn(...)` | previous gradient | Stochastic-reconfiguration-style update with momentum. |
| `adam` | `adam.get_adam_update_fn(...)` | Optax Adam state | First-order gradient update using Optax Adam. |

All three builders return an update function with the same shape:

```python
params, opt_state, data, logdict = update_fn(
    params, opt_state, data, t, subkeys
)
```

Each update function performs the core VMC step:

1. Run MCMC to refresh walker configurations.
2. Evaluate local energies on the new walkers.
3. Estimate the parameter update from local energies and log-wavefunction gradients.
4. Return updated parameters, updated optimizer state, updated walkers, and logging metrics.

The common return signature is why `train.py` can use one outer loop for all three training modes.

## 6. Compile the update step

After choosing the optimizer, `train.py` wraps the update function with `shard_map` and `jax.jit`:

```python
update_fn = jax.jit(
    shard_map(
        update_fn,
        mesh=mesh,
        in_specs=(None, None, pbatch, None, pbatch),
        out_specs=(pnone, pnone, pbatch, pnone),
        check_rep=False,
    )
)
```

The specification says that parameters and optimizer state are replicated, while walker data and random keys are sharded along the batch axis. The output mirrors the same layout: replicated parameters and optimizer state, sharded walker data, and replicated scalar metrics.

## 7. Resume or initialize from checkpoints

Before entering the training loop, `train.py` checks `args.output` for the newest `ckpt_*` file. Training checkpoints are pickled dictionaries containing:

```python
{
    "t": t,
    "data": data,
    "params": params,
    "opt_state": opt_state,
}
```

If a VMC checkpoint is found, training resumes from `t + 1`, restores walkers and parameters, and restores the optimizer state unless `--reset_opt` is set.

If `args.output` has no checkpoint but `--restore` points to another run, `train.py` calls `laqx/pretrain.py` to convert the restored parameters into a checkpoint compatible with the new output directory. This path is useful when starting a new run from a previous ansatz or checkpoint while keeping the regular VMC training loop unchanged.

## 8. Run the training loop

The main loop writes `log.csv` and advances from the resume step to `--steps`:

```text
for t in range(t_init, args.steps):
  split random keys for the walker batch
  run the compiled update function
  update adaptive compile settings if needed
  write one row to log.csv
  save ckpt_XXXXXX.npz every --save_frequency steps
```

The standard log fields are:

| Field | Meaning |
| --- | --- |
| `loss` | Mean local energy estimate. |
| `pmove` | MCMC acceptance rate. |
| `variance` | Variance of the sampled local energies. |
| `lr` | Effective learning rate reported by the optimizer. |
| `norm` | Update norm for `march` and `spring`; omitted for `adam`. |

Some Hamiltonian or fast-update paths may add auxiliary fields internally, but `train.py` removes implementation-only fields before writing the CSV.

The `tvmc` mode writes a slightly different set of columns: `loss`, `pmove`, `variance`, `norm`, and `res` (the TDVP residual ratio). The `lr` column is omitted in tVMC because tVMC uses a fixed time step set by `--lr`.

## 9. Adaptive compiled update functions

`train.py` keeps compiled update functions in an `optimizer_list` cache keyed by three shape-related values:

```text
R{args.reduce}_N{args.neighbor}_D{args.reduce2}
```

These values control padded operator-application sizes and fast-update neighbor sizes. When a step reports that a larger size is needed, the training loop updates the key, compiles a new update function, stores it in the cache, and continues. This keeps the common case fast while still allowing the code to handle larger operator neighborhoods encountered during training.

For most users, this mechanism is automatic. When tuning performance, the relevant CLI options are `--reduce`, `--reduce2`, `--pad`, `--pad2`, `--neighbor`, and `--fast_update`.

## 10. Outputs produced by training

A normal run writes two kinds of files under `--output`:

- `log.csv`: one row per training step, including loss, MCMC acceptance rate, variance, learning rate, and optimizer-specific metrics.
- `ckpt_XXXXXX.npz`: periodic pickled checkpoints containing the current step, walker data, network parameters, and optimizer state.

Although the checkpoint filenames use the `.npz` suffix, they are written with Python `pickle`. Use the checkpoint utilities or `pickle.load(...)` when reading them directly.

## Extension checklist

When modifying the training workflow, keep these contracts stable:

- New ansatzes should be registered in `laqx/networks/__init__.py` and return `network_init`, `network_apply`, and optional orbital accessors.
- New Hamiltonians should be registered in `laqx/operators/__init__.py` and return the complete `t`/`a`/`U`/`h` operator dictionary.
- New optimizers should expose the same update signature used by `march`, `spring`, and `adam` so the outer loop can remain shared.
- Anything saved in a checkpoint should remain compatible with resume logic, especially `t`, `data`, `params`, and `opt_state`.
