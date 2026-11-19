# LaQX

**La**ttice **Q**uantum simulation based on ja**X** — a JAX toolkit for neural quantum-state simulations of two-dimensional lattice models.

**Paper:** [LaQX: A JAX Toolkit for Neural Quantum State Simulations of Lattice Quantum Many-Body Models](https://arxiv.org/abs/2611.04333)

See the [full documentation](https://bytedance.github.io/laqx/) for the project overview, examples, and reference material.

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

## Citation

If you use LaQX in your research, please cite our paper:

```bibtex
@article{gu2026laqx,
  title         = {{LaQX}: A {JAX} Toolkit for Neural Quantum State Simulations of Lattice Quantum Many-Body Models},
  author        = {Gu, Yuntian and Fan, Chaohui and Li, Ruichen and Chen, Haoxiang and Fu, Weizhong and Chen, Yixiao and Lv, Dingshun and Ren, Weiluo and Xiang, Tao and Wang, Liwei and Qin, Mingpu and Li, Xiang},
  journal       = {arXiv preprint arXiv:2611.04333},
  year          = {2026},
  eprint        = {2611.04333},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  doi           = {10.48550/arXiv.2611.04333},
  url           = {https://arxiv.org/abs/2611.04333}
}
```

## License

MIT — see [LICENSE](LICENSE).
