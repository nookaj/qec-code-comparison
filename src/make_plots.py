"""
Reads Monte Carlo simulation results from sweep.csv and produces performance
comparison plots of logical error rate p_L against physical error rate p for the 3 codes,
under (1) bit-flip noise, (2) phase-flip noise, and (3) depolarising noise.
saved in figures/.
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
    "font.size": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

colors = {
    "repetition": "#1f77b4",
    "five_qubit": "#2ca02c",
    "steane": "#ff7f0e",
    "unencoded": "#000000",
}

labels = {
    "repetition": "3-qubit repetition",
    "five_qubit": "5-qubit perfect code",
    "steane": "Steane [[7,1,3]]",
    "unencoded": "Unencoded qubit",
}

noise_title = {
    "bitflip": "Bit-flip noise",
    "phaseflip": "Phase-flip noise",
    "depolarizing": "Depolarizing noise",
}

def load(path=None):
    if path is None:
        path = RESULTS_DIR / "sweep.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["p"] = float(r["p"])
            r["P_L"] = float(r["P_L"])
            rows.append(r)
    return rows

def series(rows, code, noise):
    if code == "unencoded":
        pts = sorted({(r["p"], r["P_L"]) for r in rows if r["code"] == "unencoded"})
    else:
        pts = sorted((r["p"], r["P_L"]) for r in rows if r["code"] == code and r["noise"] == noise)
    p = np.array([x[0] for x in pts])
    pl = np.array([x[1] for x in pts])
    return p, pl

def plot_per_noise(rows):
    """One figure per noise model: logical error rate vs physical error
    rate, log-log, all three codes + unencoded baseline."""
    for noise in ["bitflip", "phaseflip", "depolarizing"]:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        for code in ["unencoded", "repetition", "five_qubit", "steane"]:
            p, pl = series(rows, code, noise)
            pl = np.clip(pl, 1e-6, None)
            style = "--" if code == "unencoded" else "-"
            marker = None if code == "unencoded" else "o"
            ax.plot(p, pl, style, marker=marker, ms=4, lw=1.5,
                    color=colors[code], label=labels[code])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error probability $p$")
        ax.set_ylabel("Logical error rate $P_L$")
        ax.set_title(noise_title[noise])
        ax.legend(loc="upper left", fontsize=10.5)
        fig.savefig(FIGURES_DIR / f"plot_{noise}.png")
        plt.close(fig)

def plot_pseudothreshold(rows):
    """Bar plot: pseudo-threshold (crossing point with the
    unencoded line) for each code and noise combination."""
    from scipy.optimize import brentq

    def interp_log(p, pl):
        # log-log linear interpolation function P_L(p)
        logp, logpl = np.log10(p), np.log10(np.clip(pl, 1e-9, None))
        return logp, logpl

    results = {}
    for code in ["repetition", "five_qubit", "steane"]:
        for noise in ["bitflip", "phaseflip", "depolarizing"]:
            p, pl = series(rows, code, noise)
            diff = pl - p  # P_L(p) - p ; crossing where this changes sign
            if np.all(diff > 0):
                # code is worse than an unencoded qubit at every simulated p
                results[(code, noise)] = "never"
                continue
            sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
            if len(sign_changes) == 0:
                # code always beats unencoded in range -> threshold is beyond it
                results[(code, noise)] = ">0.4"
                continue
            i = sign_changes[-1]  # take the crossing at largest p (pseudo-threshold)
            f = lambda lp: np.interp(lp, np.log10(p), pl) - 10 ** lp
            try:
                lp_cross = brentq(f, np.log10(p[i]), np.log10(p[i + 1]))
                results[(code, noise)] = 10 ** lp_cross
            except Exception:
                results[(code, noise)] = "n/a"

    codes = ["repetition", "five_qubit", "steane"]
    noises = ["bitflip", "phaseflip", "depolarizing"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(codes))
    width = 0.25
    for j, noise in enumerate(noises):
        vals = [results[(c, noise)] for c in codes]
        vals_plot = [v if isinstance(v, float) else (0.4 if v == ">0.4" else 0.0) for v in vals]
        ax.bar(x + (j - 1) * width, vals_plot, width, label=noise_title[noise])
        for xi, v, vp in zip(x + (j - 1) * width, vals, vals_plot):
            label = f"{v:.3f}" if isinstance(v, float) else v
            ax.text(xi, vp + 0.01, label, ha="center", fontsize=9, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in codes])
    ax.set_ylabel("Pseudo-threshold $p^*$")
    ax.set_title("Pseudo-threshold comparison")
    ax.legend()
    fig.savefig(FIGURES_DIR / f"pseudothresholds.png")
    plt.close(fig)
    return results

if __name__ == "__main__":
    rows = load()
    plot_per_noise(rows)
    thresholds = plot_pseudothreshold(rows)
    print("Pseudo-thresholds:")
    for k, v in thresholds.items():
        print(" ", k, "->", v)
    print("All figures written to figures/")
