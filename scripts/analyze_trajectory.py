"""Analyze ODE trajectory properties: velocity norm and cosine similarity over time."""
import argparse
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.rcParams.update({'font.size': 12})

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
        ema = LitEma(model)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
        print("Loaded EMA weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def analyze_trajectory(sampler, batch_size, cfg_scale, sample_steps, device):
    """Run one Euler trajectory and record velocity norms + cosine sims at each step."""
    if sampler.use_cond:
        y = torch.randint(0, sampler.num_classes, (batch_size,), device=device)
    else:
        y = None

    z = torch.randn((batch_size, sampler.channels, sampler.image_size, sampler.image_size), device=device)
    dt = 1.0 / sample_steps
    t_span = torch.linspace(1, 0, sample_steps + 1, device=device)

    prev_v = None
    cos_sims = []
    v_norms = []
    delta_norms = []
    timesteps = []

    for i in tqdm(range(sample_steps), desc="Tracing trajectory"):
        t = t_span[i]
        v = sampler._velocity(z, t, y, cfg_scale)

        v_flat = v.reshape(batch_size, -1)
        v_norms.append(v_flat.norm(dim=-1).mean().item())
        timesteps.append(t.item())

        if prev_v is not None:
            pv_flat = prev_v.reshape(batch_size, -1)
            cos = F.cosine_similarity(v_flat, pv_flat, dim=-1)
            cos_sims.append(cos.mean().item())
            delta = (v_flat - pv_flat).norm(dim=-1).mean().item()
            delta_norms.append(delta)

        z = z - v * dt
        prev_v = v

    return timesteps, v_norms, cos_sims, delta_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--use_repa", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--z_dim", type=int, default=768)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_steps", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {}

    for label, ckpt_path in [("CLIP", args.checkpoint_a), ("SigLIP", args.checkpoint_b)]:
        print(f"\n{'='*50}")
        print(f"Analyzing {label}: {ckpt_path}")
        print(f"{'='*50}")
        args.checkpoint = ckpt_path
        model = load_model(args, device)
        sampler = RectifiedFlow(model)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            timesteps, v_norms, cos_sims, delta_norms = analyze_trajectory(
                sampler, args.batch_size, args.cfg_scale, args.sample_steps, device
            )

        data[label] = {
            "timesteps": timesteps,
            "v_norms": v_norms,
            "cos_sims": cos_sims,
            "delta_norms": delta_norms,
        }
        del model, sampler
        torch.cuda.empty_cache()

    # Save raw data
    with open("results/trajectory_analysis.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved results/trajectory_analysis.json")

    # --- Plot 1: Velocity norm over time ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color in [("CLIP", "#2196F3"), ("SigLIP", "#FF9800")]:
        t = data[label]["timesteps"]
        ax1.plot(t, data[label]["v_norms"], color=color, label=label, linewidth=2)
    ax1.set_xlabel("Timestep $t$")
    ax1.set_ylabel("$\\|v_\\theta(z_t, t)\\|$")
    ax1.set_title("(a) Velocity Norm Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # --- Plot 2: Cosine similarity between consecutive velocities ---
    for label, color in [("CLIP", "#2196F3"), ("SigLIP", "#FF9800")]:
        t = data[label]["timesteps"][1:]  # cos_sims has N-1 entries
        ax2.plot(t, data[label]["cos_sims"], color=color, label=label, linewidth=2)
    ax2.set_xlabel("Timestep $t$")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("(b) Trajectory Straightness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_xaxis()

    fig.suptitle("ODE Trajectory Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/trajectory_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.savefig("report/trajectory_diagnostics.pdf", bbox_inches="tight")
    print("Saved trajectory diagnostics")
    plt.close("all")


if __name__ == "__main__":
    main()
