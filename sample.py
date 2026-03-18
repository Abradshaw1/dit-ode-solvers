"""
Generate images using different ODE solvers.

Usage:
    # Euler (default), 50 steps
    python sample.py \
        --checkpoint checkpoints/encoder_a/step_99999.pth \
        --use_repa --z_dim 768 --encoder_depth 6

    # RK4, 25 steps
    python sample.py \
        --checkpoint checkpoints/encoder_a/step_99999.pth \
        --use_repa --z_dim 768 --encoder_depth 6 \
        --solver rk4 --sample_steps 25

    # Generate class grid
    python sample.py \
        --checkpoint checkpoints/encoder_a/step_99999.pth \
        --use_repa --z_dim 768 --encoder_depth 6 \
        --mode grid --n_per_class 10
"""
import argparse
import os

import torch
from torchvision.utils import make_grid, save_image

from src.dit import DiT
from src.ema import LitEma
from src.model import RectifiedFlow


def load_model(args, device):
    if args.use_repa:
        z_dims = [args.z_dim]
        model = DiT(
            input_size=32, patch_size=2, in_channels=3,
            dim=384, depth=12, num_heads=6,
            num_classes=10, learn_sigma=False,
            class_dropout_prob=0.1,
            z_dims=z_dims, encoder_depth=args.encoder_depth,
        ).to(device)
    else:
        model = DiT(
            input_size=32, patch_size=2, in_channels=3,
            dim=384, depth=12, num_heads=6,
            num_classes=10, learn_sigma=False,
            class_dropout_prob=0.1,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.use_ema and "ema" in ckpt:
        model_ema = LitEma(model)
        model_ema.load_state_dict(ckpt["ema"])
        model_ema.copy_to(model)
        print("Loaded EMA weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate images with ODE solvers")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--use_repa", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--z_dim", type=int, default=768)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "rk4", "adaptive"])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--n_per_class", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="samples")
    args = parser.parse_args()
    args.use_ema = args.use_ema and not args.no_ema

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Solver: {args.solver}, Steps: {args.sample_steps}, CFG: {args.cfg_scale}")

    model = load_model(args, device)
    sampler = RectifiedFlow(model)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if args.mode == "grid":
            print(f"Generating {args.n_per_class} images per class (10 classes)...")
            images = sampler.sample_each_class(
                args.n_per_class, cfg_scale=args.cfg_scale, sample_steps=args.sample_steps,
            )
            grid = make_grid(images, nrow=10)
            out_path = os.path.join(args.output_dir, f"grid_{args.solver}_{args.sample_steps}steps.png")
            save_image(grid, out_path)
            print(f"Saved class grid to {out_path}")
        else:
            print(f"Generating {args.batch_size} random images...")
            images = sampler.sample(
                batch_size=args.batch_size, cfg_scale=args.cfg_scale,
                sample_steps=args.sample_steps, solver=args.solver,
            )
            grid = make_grid(images, nrow=8)
            out_path = os.path.join(args.output_dir, f"samples_{args.solver}_{args.sample_steps}steps.png")
            save_image(grid, out_path)
            print(f"Saved {args.batch_size} samples to {out_path}")


if __name__ == "__main__":
    main()
