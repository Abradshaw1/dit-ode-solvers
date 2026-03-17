# ODE Solver Analysis for Rectified Flow Diffusion

Comparing numerical ODE solvers for image generation in rectified flow diffusion models.
This project was developed as a final project for a Scientific Machine Learning (SciML) graduate course.

## Background

**Rectified flow** trains a neural network to learn a velocity field $v_\theta(z, t)$ that transports samples between a noise distribution ($t=1$) and a data distribution ($t=0$). Generating an image requires solving the initial value problem (IVP):

$$\frac{dz}{dt} = -v_\theta(z, t), \quad z(1) \sim \mathcal{N}(0, I), \quad t: 1 \to 0$$

This is a **Neural ODE** — the right-hand side is parameterized by a neural network. The choice of numerical solver directly affects both the quality (FID) and computational cost (NFE) of generated images.

## Model

The velocity field is parameterized by a **Diffusion Transformer (DiT)** — a 12-layer vision transformer with 384-dim embeddings, 6 attention heads, and adaptive layer norm conditioning on timestep + class label. The model operates on 32×32 CIFAR-10 images in pixel space with patch size 2.

Two models were trained with **REPA (Representation Alignment)**, which adds a cosine-similarity alignment loss between intermediate DiT features and frozen pretrained vision encoder features:

| Checkpoint | Encoder | Embed Dim | Training Steps |
|------------|---------|-----------|----------------|
| `checkpoints/clip/step_99999.pth` | CLIP ViT-B/16 | 768 | 100K |
| `checkpoints/siglip/step_99999.pth` | SigLIP ViT-B/16 | 768 | 100K |

Training used AdamW 8-bit, lr=1e-4, batch size 128, bfloat16 mixed precision, EMA decay 0.9999, and classifier-free guidance dropout of 0.1.

## ODE Solvers Implemented

All solvers are implemented in `model.py` within the `RectifiedFlow` class:

| Solver | Order | NFE per Step | Method |
|--------|-------|-------------|--------|
| **Euler** | 1st | 1 | `sample_euler()` — Forward Euler |
| **Heun** | 2nd | 2 | `sample_heun()` — Improved Euler (predictor-corrector) |
| **RK4** | 4th | 4 | `sample_rk4()` — Classical Runge-Kutta |
| **Adaptive RK45** | 4th-5th | Variable | `sample_adaptive()` — Dormand-Prince via `torchdiffeq` |

## Setup

```bash
conda env create -f environment.yml
conda activate dit
```

All dependencies (including `torchdiffeq`) are included in `environment.yml`.

## Training

Training was performed separately. To retrain from scratch:

```bash
python train.py --use_repa --proj_coeff 0.5 --encoder_type clip --encoder_size s --encoder_depth 6
python train.py --use_repa --proj_coeff 0.5 --encoder_type siglip --encoder_size s --encoder_depth 6
```

Pretrained checkpoints should be placed in `checkpoints/clip/` and `checkpoints/siglip/`.

## Evaluating ODE Solvers

Run FID evaluation across all solvers and step counts:

```bash
# CLIP model
python evaluate_solvers.py \
    --checkpoint checkpoints/clip/step_99999.pth \
    --use_repa --z_dim 768 --encoder_depth 6 \
    --solvers euler heun rk4 \
    --step_counts 5 10 15 25 50 100 \
    --output results/solver_results_clip.json

# SigLIP model
python evaluate_solvers.py \
    --checkpoint checkpoints/siglip/step_99999.pth \
    --use_repa --z_dim 768 --encoder_depth 6 \
    --solvers euler heun rk4 \
    --step_counts 5 10 15 25 50 100 \
    --output results/solver_results_siglip.json
```

This generates 50,000 images per configuration and computes FID against CIFAR-10 training statistics. Results are saved incrementally to JSON.

## Project Structure

```
dit-ode-solvers/
├── model.py                  # RectifiedFlow with all ODE solvers + trajectory analysis
├── dit.py                    # Diffusion Transformer architecture
├── ema.py                    # Exponential moving average
├── fid_evaluation.py         # FID metric computation
├── evaluate_solvers.py       # Main evaluation script (FID vs NFE)
├── train.py                  # Training script (REPA + rectified flow)
├── repa.py                   # Encoder loading, preprocessing, alignment loss
├── evaluate_fid.py           # Single-solver FID evaluation
├── evaluate_cknna.py         # Representation alignment metrics
├── evaluate_linear_probe.py  # Linear probe evaluation
├── environment.yml           # Conda environment
└── README.md
```

## Key Results

The main deliverable is a **FID vs NFE** plot comparing solver efficiency:
- **NFE** (Number of Function Evaluations) measures computational cost
- **FID** (Fréchet Inception Distance) measures image quality (lower is better)
- Higher-order solvers (Heun, RK4) achieve better FID at low NFE budgets
- At high NFE all solvers converge, but the cost-efficiency differs

## References

- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [REPA: Representation Alignment for Generation](https://arxiv.org/abs/2402.17726)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
