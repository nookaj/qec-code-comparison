"""
Reads results from qiskit_validation.csv (written by qiskit_version.py) and
draws a scatter plot: stabilizer Monte Carlo P_L vs. real Qiskit circuit simulation P_L.
Diagonal = perfect agreement.
"""

import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

colors = {
    "3-qubit repetition code": "#1f77b4",
    "5-qubit perfect code": "#2ca02c",
    "Steane [[7,1,3]] code": "#ff7f0e",
}

markers = {"bitflip": "o", "phaseflip": "s", "depolarizing": "^"}

def main(csv_path=None, out_path=None):
    if csv_path is None:
        csv_path = RESULTS_DIR / "qiskit_validation.csv"
    if out_path is None:
        out_path = FIGURES_DIR / "qiskit_vs_montecarlo.png"
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise SystemExit(f"{csv_path} is empty -- run qiskit_version.py first.")

    pf_all = np.array([float(r["P_L_fast_mc"]) for r in rows])
    pq_all = np.array([float(r["P_L_qiskit"]) for r in rows])
    diffs = np.abs(pf_all - pq_all)
    print(f"{len(rows)} configurations loaded from {csv_path}")
    print(f"mean |P_L_qiskit - P_L_fast_mc| = {diffs.mean():.4f}")
    print(f"max  |P_L_qiskit - P_L_fast_mc| = {diffs.max():.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    hi = max(pf_all.max(), pq_all.max()) * 1.1
    lims = [0, hi]
    ax.plot(lims, lims, "--", color="black", lw=1.2, zorder=1, label="perfect agreement")

    for r in rows:
        pf = float(r["P_L_fast_mc"])
        pq = float(r["P_L_qiskit"])
        code, noise = r["code_name"], r["noise"]
        ax.scatter(pf, pq, color=colors[code], marker=markers[noise], s=80,
                   edgecolor="white", linewidth=0.7, zorder=3)

    from matplotlib.lines import Line2D
    code_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=name)
                    for name, c in colors.items()]
    noise_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=9, label=n)
                     for n, m in markers.items()]
    leg1 = ax.legend(handles=code_handles, loc="upper left", fontsize=9.5, title="Code", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=noise_handles, loc="lower right", fontsize=9.5, title="Noise model", title_fontsize=10)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Logical error rate $p_L$ (Monte Carlo)")
    ax.set_ylabel("Logical error rate $p_L$ (Qiskit circuit simulation)")
    ax.set_title("Validation: Qiskit vs. Monte Carlo")
    ax.set_aspect("equal")

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
