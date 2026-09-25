"""
Runs the full Monte Carlo sweep:
   for each code in {repetition, five_qubit, steane}
     for each noise model in {bitflip, phaseflip, depolarizing}
       for each physical error probability p in a log-spaced grid
         estimate the logical error rate P_L via vectorized Monte Carlo
Saves everything to results/sweep.csv, plus the unencoded qubit baseline.
"""

import numpy as np
import csv
import time
from pathlib import Path
from codes import get_code
from noise import sample_error_batch
from simulate import run_trials_vectorized, unencoded_logical_error_rate

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

codes = ["repetition", "five_qubit", "steane"]
noise_models = ["bitflip", "phaseflip", "depolarizing"]
p_grid = np.logspace(np.log10(0.001), np.log10(0.4), 22)

seed = 12345

def trials_for_p(p):
    """More trials for small p (rarer failure events) so relative error on
    P_L stays roughly comparable across the whole sweep."""
    n = int(np.clip(3000.0 / p, 50_000, 3_000_000))
    return n

def main():
    rng = np.random.default_rng(seed)
    rows = []
    t_start = time.time()
    for code_key in codes:
        code = get_code(code_key)
        for noise in noise_models:
            for p in p_grid:
                n_trials = trials_for_p(p)
                p_l = run_trials_vectorized(code, noise, float(p), n_trials, rng, sample_error_batch)
                rows.append({
                    "code": code_key,
                    "code_name": code.name,
                    "n_qubits": code.n,
                    "noise": noise,
                    "p": float(p),
                    "P_L": float(p_l),
                    "n_trials": n_trials,
                })
                print(f"{code.name:24s} {noise:12s} p={p:.4f}  P_L={p_l:.5f}  (N={n_trials})")
    # unencoded baseline (analytic, but also tabulated for convenience)
    for p in p_grid:
        rows.append({
            "code": "unencoded",
            "code_name": "Unencoded qubit",
            "n_qubits": 1,
            "noise": "any",
            "p": float(p),
            "P_L": float(unencoded_logical_error_rate(p)),
            "n_trials": None,
        })

    with open(RESULTS_DIR / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone in {time.time() - t_start:.1f} s. Wrote results/sweep.csv ({len(rows)} rows).")

if __name__ == "__main__":
    main()
