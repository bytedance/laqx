import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'
os.environ['NCCL_DEBUG'] = 'WARN'

import argparse
parser = argparse.ArgumentParser()

# ---- Mode & I/O ----
parser.add_argument('--mode', type=str, default="march")
parser.add_argument('--output', type=str, default="outputs/debug")
parser.add_argument('--restore', type=str, default="")
parser.add_argument('--restore_network_name', type=str, default="tensor")
parser.add_argument('--restore_layers', type=int, default=2)
parser.add_argument('--restore_hidden', type=int, default=256)
parser.add_argument('--restore_ndet', type=int, default=2)
parser.add_argument('--restore_head', type=int, default=4)
parser.add_argument('--save_frequency', type=int, default=2000)

# ---- Lattice & System ----
parser.add_argument('--model', type=str, default='hubbard')
parser.add_argument('--L1', type=int, default=4)
parser.add_argument('--L2', type=int, default=4)
parser.add_argument('--particles', type=int, default=14)
parser.add_argument('--particles_up', type=int, default=-1)
parser.add_argument('--boundary1', type=str, default="pbc")
parser.add_argument('--boundary2', type=str, default="pbc")
parser.add_argument('--polarized', action='store_true', default=False)
parser.add_argument('--use_boson', action='store_true', default=False)

# ---- Hamiltonian parameters ----
parser.add_argument('--t', type=float, default=1)
parser.add_argument('--t2', type=float, default=0)
parser.add_argument('--t3', type=float, default=0)
parser.add_argument('--U', type=float, default=8)
parser.add_argument('--V', type=float, default=0)
parser.add_argument('--V2', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--j1', type=float, default=1)
parser.add_argument('--j2', type=float, default=0)
parser.add_argument('--hm', type=float, default=0)
parser.add_argument('--htype', type=str, default="AFM")
parser.add_argument('--lambda_h', type=int, default=0)
parser.add_argument('--hv', type=float, default=0)
parser.add_argument('--flux_theta', type=float, default=0)
parser.add_argument('--flux_type', type=str, default="spin")
parser.add_argument('--marshall', action='store_true', default=False)

# ---- Network architecture ----
parser.add_argument('--network_name', type=str, default="transformer")
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--MLP_hidden', type=int, default=256)
parser.add_argument('--MLP_layers', type=int, default=2)
parser.add_argument('--num_head', type=int, default=4)
parser.add_argument('--mpsdim', type=int, default=10)
parser.add_argument('--mps_num_head', type=int, default=2)
parser.add_argument('--cutoff', type=int, default=3)
parser.add_argument('--ndet', type=int, default=4)
parser.add_argument('--activation', type=str, default="silu")
parser.add_argument('--dmax', type=int, default=1)

# ---- Training ----
parser.add_argument('--batchsize', type=int, default=4096)
parser.add_argument('--steps', type=int, default=50000)
parser.add_argument('--drop_step', type=int, default=10)
parser.add_argument('--pretrain_step', type=int, default=1000)
parser.add_argument('--mcmc_step', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--clip_el', type=float, default=5)

# ---- Optimizer ----
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--norm', type=float, default=1e-1)
parser.add_argument('--mu', type=float, default=0.95)
parser.add_argument('--beta2', type=float, default=0.995)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--damping', type=float, default=0.001)
parser.add_argument('--lr0', type=float, default=8000)
parser.add_argument('--lr_start', type=float, default=1000)
parser.add_argument('--v_init', type=float, default=1e-3)
parser.add_argument('--warmup', type=int, default=50)
parser.add_argument('--reset_opt', action='store_true', default=False)

# ---- NES (Neural Excited States) ----
parser.add_argument('--num_states', type=int, default=1)

# ---- tVMC (Time-Dependent VMC) ----
parser.add_argument('--integrator', type=str, default="")
parser.add_argument('--solver', type=str, default="cholesky")
parser.add_argument('--pinv_cutoff', type=float, default=1e-8)

# ---- GFMC (Green's Function Monte Carlo) ----
parser.add_argument('--gfmc_step', type=float, default=1e-2)

# ---- Symmetry ----
parser.add_argument('--symmetry', type=str, default='')
parser.add_argument('--particle_hole', type=str, default='')
parser.add_argument('--kx', type=int, default=0)
parser.add_argument('--ky', type=int, default=0)
parser.add_argument('--rotation', type=float, default=1)

# ---- Observables & Evaluation ----
parser.add_argument('--obs', type=str, default="energy")
parser.add_argument('--burn_in', action='store_true', default=False)

# ---- Performance tuning ----
parser.add_argument('--reduce', type=int, default=0)
parser.add_argument('--reduce2', type=int, default=0)
parser.add_argument('--pad', type=int, default=10)
parser.add_argument('--pad2', type=int, default=10)
parser.add_argument('--neighbor', type=int, default=30)
parser.add_argument('--fast_update', action='store_true', default=False)

# ---- Computation & Precision ----
parser.add_argument('--dtype', type=str, default='float')
parser.add_argument('--precision', type=str, default="fp32")
parser.add_argument('--use_x64', action='store_true', default=False)
parser.add_argument('--multi_host', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)

args = parser.parse_args()
if args.precision == "tf32":
    os.environ['NVIDIA_TF32_OVERRIDE']=""
else:
    os.environ['NVIDIA_TF32_OVERRIDE']="0"
    os.environ['JAX_DEFAULT_MATMUL_PRECISION']="float32"

import jax
if args.precision == "x64":
    jax.config.update("jax_enable_x64", True)

if args.particles_up == -1:
    args.particles_up = args.particles // 2

args.network_name = args.network_name.lower()

from laqx import gfmc, train, test, tvmc
if args.mode in ['spring', 'adam', 'march']:
    train.train(args)
elif args.mode in ['test']:
    test.train(args)
elif args.mode in ['gfmc']:
    gfmc.train(args)
elif args.mode in ['tvmc']:
    tvmc.train(args)
