# LaQX

**La**ttice **Q**uantum simulation based on ja**X** — a JAX toolkit for neural quantum-state simulations of two-dimensional lattice models.

This README only covers setup and a first command. See the full documentation for the project overview, examples, and reference material.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
pip install -c constraints.txt --upgrade "jax[cuda12_pip]==0.5.0" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install chex==0.1.90 scipy==1.13.0 optax==0.2.2 numpy==1.26.4 tqdm==4.66.5
```

## Quick Start

```bash
bash docs/examples/hubbard/16x4_pbc_U8_ace_small.sh
```

## Documentation

- [Documentation home](docs/index.md)
- [Installation guide](docs/installation.md)
- [Examples](docs/examples/index.md)
- [CLI parameters](docs/reference/parameters.md)

## License

MIT — see [LICENSE](LICENSE).
