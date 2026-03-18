"""Generate FID vs NFE and FID vs Steps plots from solver evaluation results."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

with open("results/solver_results_encoder_a.json") as f:
    results_a = json.load(f)
with open("results/solver_results_encoder_b.json") as f:
    results_b = json.load(f)

SOLVER_STYLES = {
    "euler": {"color": "#2196F3", "marker": "o", "label": "Euler (order 1)"},
    "heun":  {"color": "#FF9800", "marker": "s", "label": "Heun (order 2)"},
    "rk4":   {"color": "#4CAF50", "marker": "^", "label": "RK4 (order 4)"},
}

def get_steps_and_fids(results, solver):
    steps = sorted(results[solver], key=lambda x: int(x))
    return [int(s) for s in steps], [results[solver][s]["fid"] for s in steps]

def get_nfes_and_fids(results, solver):
    steps = sorted(results[solver], key=lambda x: int(x))
    return [results[solver][s]["nfe"] for s in steps], [results[solver][s]["fid"] for s in steps]

def plot_panel(results, title, ax, x_fn, xlabel):
    for solver, style in SOLVER_STYLES.items():
        if solver not in results:
            continue
        xs, fids = x_fn(results, solver)
        ax.plot(xs, fids, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FID ↓")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# --- Figure 1: FID vs NFE (side-by-side) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_panel(results_a, "CLIP-aligned (Encoder A)", ax1, get_nfes_and_fids, "Number of Function Evaluations (NFE)")
plot_panel(results_b, "SigLIP-aligned (Encoder B)", ax2, get_nfes_and_fids, "Number of Function Evaluations (NFE)")
ax1.set_ylim(14, 75); ax2.set_ylim(14, 75)
fig.suptitle("FID vs NFE: ODE Solver Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fid_vs_nfe.png", dpi=200, bbox_inches="tight")
plt.savefig("report/fid_vs_nfe.pdf", bbox_inches="tight")
print("Saved fid_vs_nfe")

# --- Figure 2: FID vs NFE zoomed (NFE <= 100) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_panel(results_a, "CLIP-aligned (Encoder A)", ax1, get_nfes_and_fids, "Number of Function Evaluations (NFE)")
plot_panel(results_b, "SigLIP-aligned (Encoder B)", ax2, get_nfes_and_fids, "Number of Function Evaluations (NFE)")
ax1.set_xlim(0, 110); ax2.set_xlim(0, 110)
ax1.set_ylim(14, 40); ax2.set_ylim(14, 40)
fig.suptitle("FID vs NFE: Low-NFE Regime (≤100)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fid_vs_nfe_zoomed.png", dpi=200, bbox_inches="tight")
plt.savefig("report/fid_vs_nfe_zoomed.pdf", bbox_inches="tight")
print("Saved fid_vs_nfe_zoomed")

# --- Figure 3: Side-by-side NFE vs Steps (CLIP model) ---
# This is the key comparison: same data, two x-axes, curve ordering FLIPS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plot_panel(results_a, "(a) FID vs NFE (computational cost)", ax1, get_nfes_and_fids, "Number of Function Evaluations (NFE)")
plot_panel(results_a, "(b) FID vs Sampling Steps", ax2, get_steps_and_fids, "Number of Solver Steps")
ax1.set_ylim(14, 40); ax2.set_ylim(14, 40)
ax1.set_xlim(0, 110); ax2.set_xlim(0, 110)
# Remove duplicate y-label on right panel
ax2.set_ylabel("")
fig.suptitle("Steps ≠ Cost: Higher-order methods reduce steps but not NFE efficiency",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fid_steps_vs_nfe.png", dpi=200, bbox_inches="tight")
plt.savefig("report/fid_steps_vs_nfe.pdf", bbox_inches="tight")
print("Saved fid_steps_vs_nfe")

plt.close("all")
print("Done!")
