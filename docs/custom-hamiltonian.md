# Custom Hamiltonians

This guide explains the LaQX operator contract and how to add a new Hamiltonian using the same format as the built-in models.

## What a Hamiltonian module must provide

Create a module under `laqx/operators/hamiltonian/` with a constructor named `make_<model>(args)`. The constructor must return all four standard groups:

```python
def make_my_model(args):
    return {
        "t": get_operators_and_weights_t(args),
        "a": get_operators_and_weights_a(args),
        "U": get_operators_and_weights_U(args),
        "h": get_operators_and_weights_h(args),
    }
```

Hamiltonians and observables live under `laqx/operators/hamiltonian/` and are registered in `laqx/operators/__init__.py`.

Return `(None, None)` for unused groups. Do not omit keys, because the local-energy and sampling code expect the complete `t`/`a`/`U`/`h` contract.

## Operator groups

Each active group is a pair `(operators, weights)` of JAX arrays. `operators` contains integer site indices and `weights` contains real or complex coefficients.

LaQX represents Hamiltonians of the form:

$$
H = \sum_{ij} t_{ij} c_i^\dagger c_j
  + \sum_{ijkl} a_{ijkl} c_i^\dagger c_j^\dagger c_k c_l
  + \sum_{ij} U_{ij} n_i n_j
  + \sum_i h_i n_i .
$$

| Group | Shape | Interpretation |
| --- | --- | --- |
| `t` | `(n_terms, 2)` | One particle moves from source to target. |
| `a` | `(n_terms, 4)` | Two particles move from the first two sites to the last two sites. |
| `U` | `(n_terms, 2)` | Diagonal two-site occupation contribution. |
| `h` | `(n_terms, 1)` | Diagonal one-site occupation contribution. |

Use zero-based row-major site indexing:

```python
site = i * args.L2 + j
```

For spinful fermions, use the first `L1 * L2` sites for spin up and add an offset of `L1 * L2` for spin down.

## Boundary convention

The built-in square-lattice models use:

```python
if (i < args.L1 - 1) or (args.boundary1 == "pbc"):
    # add first-direction bond

if (j < args.L2 - 1) or (args.boundary2 == "pbc"):
    # add second-direction bond
```

Only the exact string `pbc` enables wrapping. Use modulo arithmetic only after deciding that the bond is physically present.

## Example: Haldane model

The built-in Haldane model in `laqx/operators/hamiltonian/haldane.py` is a useful example because its honeycomb unit cell is effectively `2 x 1`: the parity of the row index chooses which sublattice geometry to use. LaQX does not need a separate unit-cell abstraction for this case. The Hamiltonian can still be encoded with the same row-major site index and the same four operator groups.

The excerpt below shows the nearest-neighbor hopping and density interaction parts. The full implementation also adds optional complex second- and third-neighbor hoppings when `args.t2` or `args.t3` are set.

```python
import jax.numpy as jnp
import numpy as np


def site(i, j, args):
    return (i % args.L1) * args.L2 + (j % args.L2)


def get_operators_and_weights_t(args):
    operators = []
    weights = []

    for i in range(args.L1):
        for j in range(args.L2):
            start = site(i, j, args)
            end1 = site(i + 1, j, args)

            if (i != args.L1 - 1) or (args.boundary1 == "pbc"):
                operators += [[start, end1], [end1, start]]
                weights += [
                    args.t,
                    args.t,
                ]

            # The 2 x 1 unit cell appears as an even/odd row pattern.
            # Even rows have an additional nearest neighbor at (i - 1, j + 1).
            if (i % 2) == 0:
                end2 = site(i - 1, j + 1, args)
                operators += [[start, end2], [end2, start]]
                weights += [
                    args.t,
                    args.t,
                ]

    return jnp.array(operators), jnp.array(weights)


def get_operators_and_weights_a(args):
    return None, None


def get_operators_and_weights_U(args):
    operators = []
    weights = []

    for i in range(args.L1):
        for j in range(args.L2):
            start = site(i, j, args)

            end1 = site(i + 1, j, args)
            operators.append([start, end1])
            weights.append(args.V)

            if (i % 2) == 0:
                end2 = site(i - 1, j + 1, args)
                operators.append([start, end2])
                weights.append(args.V)

    return jnp.array(operators), jnp.array(weights)


def get_operators_and_weights_h(args):
    return None, None


def make_haldane(args):
    return {
        "t": get_operators_and_weights_t(args),
        "a": get_operators_and_weights_a(args),
        "U": get_operators_and_weights_U(args),
        "h": get_operators_and_weights_h(args),
    }
```

For complex hopping phases, put complex numbers in `weights` and run with `--dtype complex`.

## Register the model

Import your constructor in `laqx/operators/__init__.py`:

```python
from .hamiltonian.my_model import make_my_model
```

Then add a branch in `get_operator(args, inference=False)`:

```python
elif args.model == "my_model":
    operator = make_my_model(args)
```

After registration, run it with:

```bash
python main.py --model my_model --mode march ...
```

## Add parameters only when necessary

All CLI options live in `main.py`. If your model needs a new coupling, add it there:

```python
parser.add_argument("--phi", type=float, default=0.0)
```

Then access it as `args.phi` in your model module. Prefer reusing existing options such as `--t`, `--U`, `--V`, `--alpha`, or `--flux_theta` when their meaning is clear for your model.

## How the contract is used at runtime

The local-energy path is:

```text
operators.operator_provider(...)
  -> get_operator(args, inference)
  -> make_operator(network, operator, args)
  -> apply_operator.py combines t/a/U/h contributions
```

The sampling path is:

```text
operators.mcmc_provider(...)
  -> get_operator(args)
  -> mcmc.make_mcmc_step(...)
```

## MCMC proposals are automatic

You do not need to write a sampler for a new Hamiltonian. `mcmc.py` builds proposals from the off-diagonal `t` and `a` groups returned by your model. Diagonal `U` and `h` terms contribute to the local energy but do not generate moves.

If your Hamiltonian has no off-diagonal `t` or `a` entries, the current MCMC provider cannot propose moves and raises `NotImplementedError`.

## Checklist

- Return all four keys: `t`, `a`, `U`, `h`.
- Return `(None, None)` for inactive groups.
- Add Hermitian-conjugate hopping directions explicitly when the physical Hamiltonian requires them.
- Use the correct spin offset for spinful models.
- Respect `boundary1` and `boundary2` before wrapping coordinates.
- Use complex weights together with `--dtype complex` for complex Hamiltonians.
- Register the model in `laqx/operators/__init__.py`.
- Add new CLI options in `main.py` only if existing options are insufficient.

## Example: ground-state Haldane run

The built-in Haldane model is a useful end-to-end check for custom complex Hamiltonians. The script below follows the same ground-state VMC style as the Hofstadter example in `docs/examples/tvmc/hof_8x6_obc.sh`, but switches the model to `haldane` and enables complex second-neighbor hopping with `--t2`.

```bash
python main.py \
 --output outputs/haldane/8_6_N4_pbc_V2_t2/ace_small_N5e-1\
 --L1 8\
 --L2 6\
 --particles 4\
 --particles_up 4\
 --t 1\
 --t2 0.2\
 --V 2\
 --model haldane\
 --dtype complex\
 --steps 10000\
 --network_name ace\
 --boundary1 pbc\
 --boundary2 pbc\
 --save_frequency 2000\
 --use_x64\
 --mcmc_step 40\
 --mode march\
 --norm 5e-1\
 --lr_start 1000\
 --lr0 4000\
 --ndet 1 \
 --hidden 128\
 --layers 12\
 --MLP_hidden 256\
 --MLP_layers 1\
 --reduce 100\
 --pad 5\
 --seed 100\
 --precision tf32\
 --batchsize 4096\
 --polarized
```

The important Haldane-specific choices are `--model haldane`, `--dtype complex`, and a nonzero `--t2`. The `--polarized` setting makes this a single-spin-channel calculation, matching the spinless Haldane operator convention used by the built-in implementation.
