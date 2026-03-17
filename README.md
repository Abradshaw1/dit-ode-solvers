# ODE Solver Analysis for Rectified Flow Diffusion

This project analyzes how different numerical ODE solvers affect image generation quality in rectified flow diffusion models. Rectified flow learns a velocity field $v_\theta(z, t)$ and generates images by solving the ODE $dz/dt = -v_\theta(z, t)$ from $t=1$ (noise) to $t=0$ (data).

## Solvers Implemented

- **Euler** (1st order, 1 NFE/step)
- **Heun / Improved Euler** (2nd order, 2 NFE/step)
- **RK4** (4th order, 4 NFE/step)
- **Adaptive Dormand-Prince (RK45)** via `torchdiffeq` (variable NFE)

## Setup

```bash
conda env create -f environment.yml
conda activate dit
pip install torchdiffeq
```

## Evaluate Solvers

```bash
python evaluate_solvers.py \
    --checkpoint /home/aidan/DiT-REPA/checkpoints/20260306_190020/step_99999.pth \
    --use_repa --z_dim 768 --encoder_depth 6 \
    --solvers euler heun rk4 adaptive \
    --step_counts 5 10 15 25 50 100
```

## Base Model

The underlying model is a 12-layer Diffusion Transformer (DiT) trained on CIFAR-10 with REPA (Representation Alignment) using rectified flow. Checkpoints are stored in the companion repo `DiT-REPA`.
