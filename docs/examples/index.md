# Examples

This section contains end-to-end LaQX workflows. Each page explains the physics target and the exact `python main.py ...` commands used by the corresponding scripts under `docs/examples/`.

The shell scripts are still kept in the repository for reproducibility, but each example page also expands the script into explicit commands so that every model parameter, optimizer choice, boundary condition, and ansatz setting is visible.

## Example map

| Page | Physics task | Hamiltonian / objective | Main ansatz |
| --- | --- | --- | --- |
| [Hubbard Model](hubbard.md) | Ground-state VMC | Hubbard model | `tensor`, `ace`, `scale` |
| [Spin Model](spin.md) | Frustrated quantum magnet ground state | $J_1$-$J_2$ Heisenberg model | `cnn_mps` |
| [Altermagnetic Hubbard Model](alter-hubbard.md) | Ground-state VMC and momentum-distribution measurement | Altermagnetic Hubbard model | `tensor`, `ace` |
| [Hofstadter Model](hofstadter.md) | Many-body Chern-number workflow by flux threading | Interacting Hofstadter model with polar observable | `ace` |
| [Excited States (NES)](nes.md) | Simultaneous low-lying-state optimization | Multi-state Hofstadter VMC | `ace_nes` |
| [Time Evolution (tVMC)](tvmc.md) | Time-dependent variational principle evolution | Projected dynamics on the variational manifold | `ace`, `ace_peft` |

## How to read the commands

Each command has the same structure:

```bash
python main.py \
    --mode <march|spring|adam|test|gfmc|tvmc> \
    --model <hamiltonian-name> \
    --network_name <ansatz-name> \
    --output <checkpoint-and-log-directory> \
    ...
```

Important groups of arguments:

- `--model`, `--L1`, `--L2`, `--particles`, and boundary flags define the Hilbert space and Hamiltonian geometry.
- Couplings such as `--U`, `--V`, `--t2`, `--alpha`, `--j1`, and `--j2` define the physics problem.
- `--network_name`, `--hidden`, `--layers`, and related flags define the neural ansatz.
- `--mode adam`, `--mode march`, `--mode spring`, `--mode test`, `--mode gfmc`, and `--mode tvmc` choose pretraining, VMC optimization, observable measurement, GFMC projection, or time evolution.
- `--restore` reuses a previous checkpoint as initialization for a later stage.
