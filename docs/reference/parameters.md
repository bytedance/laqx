# CLI Parameters

All command-line options are defined in `main.py`.

## Mode and I/O

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--mode` | str | `march` | Run mode. Supported dispatch values are `march`, `spring`, `adam`, `test`, `gfmc`, and `tvmc`. |
| `--output` | str | `outputs/debug` | Output directory. `march`/`spring`/`adam` and `tvmc` create it when needed; `test` and `gfmc` expect an existing checkpoint there. |
| `--restore` | str | `""` | Checkpoint directory used in pretrain or tVMC. |
| `--save_frequency` | int | `2000` | Checkpoint interval for training. |
| `--restore_network_name` | str | `tensor` | Network name used by the pretrain. |
| `--restore_layers` | int | `2` | Layer count for the auxiliary restored network in the pretrain. |
| `--restore_hidden` | int | `256` | Hidden width for the auxiliary restored network in the pretrain. |
| `--restore_ndet` | int | `2` | Determinant count for the auxiliary restored network in the pretrain. |
| `--restore_head` | int | `4` | Attention head count for the auxiliary restored network in the pretrain. |

## Lattice and particle sector

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | str | `hubbard` | Hamiltonian/model provider. Registered names are `hubbard`, `hubbard_alter`, `hofstadter`, `tri_hofstadter`, `haldane`, and `spin`. |
| `--L1` | int | `4` | Lattice size in the first direction. |
| `--L2` | int | `4` | Lattice size in the second direction. |
| `--particles` | int | `14` | Total number of occupied entries. For spinful fermions this is up plus down particles. |
| `--particles_up` | int | `-1` | Number of up-spin particles. If left at `-1`, it is set to `particles // 2`. |
| `--boundary1` | str | `pbc` | First-direction boundary condition. The code checks for the exact string `pbc`; other strings act as open boundaries in most models. |
| `--boundary2` | str | `pbc` | Second-direction boundary condition. The code checks for the exact string `pbc`; other strings act as open boundaries in most models. |
| `--polarized` | flag | `False` | Use a single physical spin channel in supported polarized-fermion workflows. |
| `--use_boson` | flag | `False` | Use bosonic/spin-style occupation handling in supported paths, mainly spin examples. |

## Hamiltonian parameters

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--t` | float | `1` | Hopping amplitude for `hubbard`, `hubbard_alter`, and `haldane`. The square Hofstadter model uses unit hopping and ignores `--t`. |
| `--t2` | float | `0` | Model-specific longer-range hopping: diagonal hopping in `hubbard`/`hubbard_alter`, complex second-neighbor hopping in `haldane`. |
| `--t3` | float | `0` | Additional longer-range hopping in `hubbard` and `haldane`. |
| `--U` | float | `8` | Onsite interaction in `hubbard`, `hubbard_alter`, and `tri_hofstadter`. |
| `--V` | float | `0` | Nearest-neighbor density interaction in `hofstadter` and `haldane`. |
| `--V2` | float | `0` | Parsed but not used by the current built-in Hamiltonians. |
| `--alpha` | float | `0` | Background flux density in the square Hofstadter model. |
| `--j1` | float | `1` | Nearest-neighbor exchange in the spin model. |
| `--j2` | float | `0` | Next-nearest-neighbor exchange in the spin model. |
| `--hm` | float | `0` | Hubbard pinning-field strength. When zero, no Hubbard pinning field is added. |
| `--htype` | str | `AFM` | Hubbard pinning-field pattern selector. The code checks substrings such as `spin`, `neel`, `AFM`, `hole`, `board`, `hori`, `left`, and `obc`. |
| `--lambda_h` | int | `0` | Period/spacing parameter used by several Hubbard pinning-field patterns. |
| `--hv` | float | `0` | Site-0 potential in the square Hofstadter model. |
| `--flux_theta` | float | `0` | Threaded-flux parameter in Hofstadter, Haldane, and triangular-Hofstadter workflows. |
| `--flux_type` | str | `spin` | Triangular-Hofstadter flux convention. `spin` uses opposite flux signs for the two spin sectors; other strings use the same sign. |
| `--marshall` | flag | `False` | Apply the Marshall sign convention to nearest-neighbor spin-exchange off-diagonal terms. |

## Network architecture

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--network_name` | str | `transformer` | Network provider name. Registered names include `transformer`, `nnb`, `tensor`, `scale`, `ace`, `ace_peft`, `cnn_mps`, `tensor_nes`, and `ace_nes`. |
| `--hidden` | int | `256` | Hidden channel width used by most neural ansatzes. |
| `--layers` | int | `4` | Number of layers, attention blocks, convolution blocks, or MPS-generating blocks depending on the network. |
| `--MLP_hidden` | int | `256` | Hidden width for MLP blocks where present. |
| `--MLP_layers` | int | `2` | Number of MLP layers where present. |
| `--num_head` | int | `4` | Attention head count for `transformer` and restore/pretrain auxiliary networks. |
| `--mpsdim` | int | `10` | MPS bond dimension for `cnn_mps`. |
| `--mps_num_head` | int | `2` | Number of MPS heads for `cnn_mps`. |
| `--cutoff` | int | `3` | Convolution/local-neighborhood cutoff used by ACE/SCALE/CNN-MPS-style networks. |
| `--ndet` | int | `4` | Number of determinant channels or determinant blocks. |
| `--activation` | str | `silu` | Activation function key. Supported values are `silu`, `tanh`, and `poly`. |
| `--dmax` | int | `1` | SCALE fast-update mode. `0` uses a global recomputation path; nonzero values use local-neighborhood updates. |

## Training, sampling, and optimizers

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batchsize` | int | `4096` | Number of walkers/samples per iteration. |
| `--steps` | int | `50000` | Number of iterations for training, testing, GFMC propagation, or tVMC evolution, depending on `--mode`. |
| `--drop_step` | int | `10` | Number of MCMC burn/drop steps in `test`, `gfmc`, and `--burn_in` workflows. |
| `--pretrain_step` | int | `1000` | Number of steps used by the training restore/pretrain conversion path. |
| `--mcmc_step` | int | `256` | Number of Metropolis proposal updates per sampling call. |
| `--seed` | int | `0` | JAX PRNG seed. |
| `--clip_el` | float | `5` | Local-energy clipping scale in the VMC optimizers. |
| `--lr` | float | `1e-4` | Learning rate for Adam and tVMC updates. |
| `--norm` | float | `1e-1` | Update-norm/trust-region scale for MARCH and SPRING-style VMC updates. |
| `--mu` | float | `0.95` | Momentum/averaging coefficient in optimizers that use momentum-like state. |
| `--beta2` | float | `0.995` | Adam second-moment coefficient. |
| `--eps` | float | `1e-3` | Numerical stabilizer used by VMC optimizer code. |
| `--damping` | float | `0.001` | Diagonal damping for linear solves in SR/TDVP-style updates. |
| `--lr0` | float | `8000` | Decay scale in the VMC learning-rate schedule. |
| `--lr_start` | float | `1000` | Step offset before VMC learning-rate decay begins. |
| `--v_init` | float | `1e-3` | Initial velocity-like optimizer state for MARCH. |
| `--warmup` | int | `50` | Warmup length in the MARCH learning-rate schedule. |
| `--reset_opt` | flag | `False` | When resuming a VMC or tVMC checkpoint, ignore the saved optimizer state and reinitialize it. |
| `--burn_in` | flag | `False` | Run `drop_step` MCMC-only iterations before the main training or tVMC loop. |

## NES, tVMC, GFMC, symmetry, and observables

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--num_states` | int | `1` | Number of states for NES networks and NES MCMC/data layout. Values greater than one require an NES-compatible network such as `ace_nes` or `tensor_nes`. |
| `--integrator` | str | `""` | tVMC integrator. Implemented values are `rk2` and `rk4`; examples use `rk4`. |
| `--solver` | str | `cholesky` | TDVP linear solver. `cholesky` uses a damped Cholesky solve; other strings use an eigendecomposition-based pseudoinverse with a smooth cutoff (`smooth_solve`). |
| `--pinv_cutoff` | float | `1e-8` | Pseudoinverse smoothing cutoff for the non-`cholesky` TDVP solver path. |
| `--gfmc_step` | float | `1e-2` | GFMC imaginary-time step scale. |
| `--symmetry` | str | `""` | Symmetry projection key consumed by network utilities. Common example values are `T` and `D4`. |
| `--particle_hole` | str | `""` | Enables particle-hole-specific handling when nonempty, mainly in Hubbard hopping/field signs and symmetry utilities. |
| `--kx` | int | `0` | Translation momentum quantum number in direction 1 for translation-symmetry projections. |
| `--ky` | int | `0` | Translation momentum quantum number in direction 2 for translation-symmetry projections. |
| `--rotation` | float | `1` | Rotation-sector weight used by some symmetry utilities. |
| `--obs` | str | `energy` | Observable selector. `energy` uses the selected Hamiltonian; `fermi`, `localo`, `density`, and `polar` select measurement operators in supported inference/tVMC workflows. |

## Performance and precision

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--reduce` | int | `0` | Static padding/gather size for one-particle off-diagonal operator applications. When nonzero, the code can auto-adjust it upward using `--pad`. |
| `--reduce2` | int | `0` | Static padding/gather size for two-particle off-diagonal operator applications. When nonzero, the code can auto-adjust it upward using `--pad2`. |
| `--pad` | int | `10` | Padding granularity used when auto-adjusting `--reduce`. |
| `--pad2` | int | `10` | Padding granularity used when auto-adjusting `--reduce2`. |
| `--neighbor` | int | `30` | Initial neighbor budget for cached fast-update paths. The code can auto-adjust it when `--fast_update` is enabled. |
| `--fast_update` | flag | `False` | Enable cached determinant/local-update paths where implemented, currently mainly `tensor` and `scale`. |
| `--dtype` | str | `float` | Network/operator scalar type selector. Use `complex` for complex Hamiltonians such as Hofstadter/Haldane examples. |
| `--precision` | str | `fp32` | Startup precision mode. `tf32` leaves NVIDIA TF32 enabled; `x64` enables global JAX x64 at startup; other values disable TF32 and request float32 matmul precision. |
| `--use_x64` | flag | `False` | Enable JAX x64 inside run-mode code after checkpoint loading. |
| `--multi_host` | flag | `False` | Initialize the JAX distributed runtime. |
| `--debug` | flag | `False` | In training mode, start CUDA profiling at step 3 and stop it after training. |

## Checkpoint and output notes

Checkpoint files are pickled Python payloads named like `ckpt_002000.npz`. The checkpoint helper selects the reverse-lexicographically latest filename containing `ckpt_`, so the zero-padded names written by LaQX sort correctly.

Training and tVMC checkpoints contain `t`, `data`, `params`, and usually `opt_state`. Pretrain-style checkpoints use `t = -1` and may omit optimizer state. Training writes `log.csv`; `test` writes text or pickle results depending on `--obs` and `--num_states`; `gfmc` writes `gfmc.csv`; tVMC with `--obs density` writes `density.npz`.
