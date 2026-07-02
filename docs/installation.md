# Installation

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- CUDA 12 toolkit

## Install JAX

```bash
# CUDA 12
pip install -c constraints.txt --upgrade "jax[cuda12_pip]==0.5.0" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CPU-only
pip install --upgrade "jax[cpu]==0.5.0"
```

## Install Dependencies

```bash
pip install -c constraints.txt chex==0.1.90 scipy==1.13.0 optax==0.2.2 numpy==1.26.4 tqdm==4.66.5
```

## Verify Installation

```bash
python -c "import jax; print(jax.devices())"
```

You should see your GPU(s) listed.
