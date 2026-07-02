# LaQX

**LaQX** (**La**ttice **Q**uantum simulation based on ja**X**) is a JAX-based neural quantum states toolkit for two-dimensional lattice many-body problems. It combines variational Monte Carlo, neural-network wavefunction ansatzes, lattice Hamiltonians, and distributed training utilities in one research-oriented codebase.

## What LaQX does

LaQX represents a many-body wavefunction with neural-network ansatzes, samples electron or spin configurations with Markov-chain Monte Carlo, evaluates local energies from lattice Hamiltonians, and optimizes the parameters using VMC-style update rules.

## Variational Monte Carlo in brief

Variational Monte Carlo (VMC) is a stochastic method for approximating the ground state of a quantum many-body Hamiltonian. Instead of storing the full wavefunction over an exponentially large Hilbert space, VMC chooses a parameterized trial state $\psi_\theta(x)$, where $x$ is a many-body configuration such as a spin pattern or an electron occupation pattern. The variational principle states that the expected energy

$$
E(\theta) = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}{\langle \psi_\theta | \psi_\theta \rangle}
$$

is an upper bound to the true ground-state energy. VMC therefore turns the quantum problem into an optimization problem: adjust $\theta$ until this energy estimate is minimized.

The Monte Carlo part enters by sampling configurations from the probability distribution

$$
p_\theta(x) = \frac{|\psi_\theta(x)|^2}{\sum_{x'} |\psi_\theta(x')|^2},
$$

usually with a Markov chain. On these samples, the energy can be estimated through the local energy

$$
E_{\mathrm{loc}}(x) = \frac{(H\psi_\theta)(x)}{\psi_\theta(x)},
$$

whose sample average gives $E(\theta)$. In practice a VMC iteration alternates between sampling configurations, evaluating local energies and gradients, and updating the wavefunction parameters with optimizers such as stochastic reconfiguration or Adam.

## Neural-network ansatzes

Neural quantum states use a neural network as the ansatz for $\psi_\theta(x)$ or $\log \psi_\theta(x)$. Compared with traditional fixed-form wavefunctions, neural networks can learn flexible correlation patterns directly from data generated during Monte Carlo sampling. For lattice fermion problems, LaQX combines neural networks with determinant or backflow-style structures so that the ansatz can encode both fermionic antisymmetry and many-body correlations.

In LaQX, the ansatz is selected with `--network_name` and implemented under `laqx/networks/`. During training, the ansatz provides amplitudes for sampled configurations; `laqx/operators/` applies the Hamiltonian to compute local energies; and `laqx/utils/` updates the network parameters. This is the core neural VMC workflow used by the `march`, `spring`, and `adam` modes.

At a high level, a training run follows this path:

```text
main.py
  -> select mode: march / spring / adam / tvmc / gfmc / test
  -> build a network ansatz from laqx/networks/
  -> build a Hamiltonian or observable from laqx/operators/
  -> sample configurations with MCMC
  -> update parameters with laqx/utils/ optimizers
  -> write checkpoints and log.csv to the output directory
```

See [Code Structure](code-structure.md) for a guided tour of the repository.

## Key features

- **Neural quantum states for 2D lattice systems**: Hubbard, Hofstadter, Haldane, Heisenberg/J1-J2, and related lattice models.
- **Multiple ansatz families**: Transformer, ACE, SCALE and CNN-MPS.
- **Several simulation modes**: Ground-state VMC training, Neural Excited States (NES), time-dependent VMC (tVMC), Green's Function Monte Carlo (GFMC), and evaluation/test mode.
- **JAX-first implementation**: Vectorized operators, JIT compilation, sharded batches, mixed precision, and multi-host execution support.
- **Ready-to-run experiments**: Shell scripts under `docs/examples/` reproduce common Hubbard, Hofstadter, spin, NES, and tVMC workflows.

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Command-line entry point. Parses arguments and dispatches to the selected run mode. |
| `laqx/train.py` | Main VMC training loop for `march`, `spring`, and `adam`. |
| `laqx/tvmc.py` | Time-dependent VMC loop. |
| `laqx/gfmc.py` | Green's Function Monte Carlo loop. |
| `laqx/test.py` | Checkpoint evaluation and observable measurement. |
| `laqx/networks/` | Wavefunction ansatz implementations and reusable neural-network blocks. |
| `laqx/operators/` | Hamiltonians, observables, local-energy evaluation, and MCMC moves. |
| `laqx/utils/` | Optimizers, TDVP utilities, checkpointing, and distributed-runtime helpers. |
| `docs/examples/` | Runnable experiment scripts and generated figures. |

## Quick example

```bash
# Train a Hubbard model on a 16x4 lattice with the ACE ansatz.
bash docs/examples/hubbard/16x4_pbc_U8_ace_small.sh
```

For environment setup, see [Installation](installation.md). For complete workflows, see the [Examples](examples/index.md).

## License

MIT License — see the repository `LICENSE` file for details.
