"""
Evaluate FID scores across different ODE solvers and step counts.

Usage:
    python evaluate_solvers.py \
        --checkpoint checkpoints/encoder_a/step_99999.pth \
        --use_repa --z_dim 768 --encoder_depth 6 \
        --solvers euler heun rk4 \
        --step_counts 5 10 15 25 50 100 \
        --output results/solver_results_encoder_a.json
"""
import argparse
import json
import os
import time

import torch
import torchvision
from torchvision import transforms as T

from src.dit import DiT
from src.ema import LitEma
from src.model import RectifiedFlow
from src.fid_evaluation import FIDEvaluation


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
        print("Loaded model weights")
    else:
        model.load_state_dict(ckpt)

    if "step" in ckpt:
        print(f"Checkpoint step: {ckpt['step']}")

    model.eval()
    return model


class SolverFIDEvaluation(FIDEvaluation):
    """FIDEvaluation that accepts solver and step count."""

    @torch.inference_mode()
    def fid_score(self, cfg_scale=5.0, sample_steps=25, solver="euler"):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()

        from src.fid_evaluation import num_to_groups
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )

        from tqdm.auto import tqdm
        import numpy as np
        from pytorch_fid.fid_score import calculate_frechet_distance

        for batch in tqdm(batches):
            fake_samples = self.sampler.sample(
                batch_size=batch, cfg_scale=cfg_scale,
                sample_steps=sample_steps, solver=solver,
            )
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)

        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).float().cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--use_repa", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--z_dim", type=int, default=384)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/cifar10")
    parser.add_argument("--solvers", nargs="+", default=["euler", "heun", "rk4"])
    parser.add_argument("--step_counts", nargs="+", type=int, default=[5, 10, 15, 25, 50, 100])
    parser.add_argument("--output", type=str, default="results/solver_results.json")
    args = parser.parse_args()
    args.use_ema = args.use_ema and not args.no_ema

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args, device)
    sampler = RectifiedFlow(model)

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dl = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8,
    )
    train_dl = cycle(train_dl)

    fid_eval = SolverFIDEvaluation(
        args.batch_size, train_dl, sampler,
        num_fid_samples=args.num_fid_samples,
    )

    results = {}

    for solver in args.solvers:
        results[solver] = {}
        for steps in args.step_counts:
            nfe = steps
            if solver == "heun":
                nfe = steps * 2
            elif solver == "rk4":
                nfe = steps * 4

            print(f"\n{'='*60}")
            print(f"Solver: {solver}, Steps: {steps}, NFE: {nfe}")
            print(f"{'='*60}")

            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fid = fid_eval.fid_score(
                    cfg_scale=args.cfg_scale,
                    sample_steps=steps,
                    solver=solver,
                )
            elapsed = time.time() - t0

            results[solver][str(steps)] = {
                "fid": float(fid),
                "nfe": nfe,
                "wall_time_sec": round(elapsed, 1),
            }
            print(f"FID: {fid:.4f} | NFE: {nfe} | Time: {elapsed:.1f}s")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for solver in results:
        for steps, data in results[solver].items():
            print(f"{solver:>8s} | steps={steps:>3s} | NFE={data['nfe']:>4d} | FID={data['fid']:.2f} | {data['wall_time_sec']}s")


if __name__ == "__main__":
    main()
