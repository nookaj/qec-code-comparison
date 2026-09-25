"""
Draws a "stabilizer tableau" figure for each QEC code to show its stabilizer structure: rows = stabilizer
generators (and logical operators), columns = physical qubits, cells show the Pauli operator acting on that qubit.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from codes import get_code

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

colors = {
    "I": "#EAEAEA",
    "X": "#4C9AE8",
    "Y": "#9B59B6",
    "Z": "#E8744C",
}

def pauli_grid(xvec, zvec):
    labels = []
    for x, z in zip(xvec, zvec):
        if x and z:
            labels.append("Y")
        elif x:
            labels.append("X")
        elif z:
            labels.append("Z")
        else:
            labels.append("I")
    return labels

def draw_tableau(code, row_labels, rows, filename, title):
    n = code.n
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 0.65 * n_rows + 1.2))
    for r, (xvec, zvec) in enumerate(rows):
        labels = pauli_grid(xvec, zvec)
        for c, lab in enumerate(labels):
            ax.add_patch(mpatches.Rectangle((c, n_rows - r - 1), 1, 1,
                                             facecolor=colors[lab],
                                             edgecolor="white", linewidth=2))
            ax.text(c + 0.5, n_rows - r - 0.5, lab, ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if lab != "I" else "#888888")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([f"$q_{{{i}}}$" for i in range(n)], fontsize=12)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels[::-1], fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rep = get_code("repetition")
    draw_tableau(
        rep,
        ["$Z_1Z_2$", "$Z_2Z_3$", "$\\bar{X}$", "$\\bar{Z}$"],
        rep.stabilizers + [rep.Xbar, rep.Zbar],
        str(FIGURES_DIR / "tableau_repetition.png"),
        "3-qubit repetition code: stabilizers and logicals",
    )

    five = get_code("five_qubit")
    draw_tableau(
        five,
        ["$S_1$", "$S_2$", "$S_3$", "$S_4$", "$\\bar{X}$", "$\\bar{Z}$"],
        five.stabilizers + [five.Xbar, five.Zbar],
        str(FIGURES_DIR / "tableau_five_qubit.png"),
        "5-qubit perfect code: stabilizers and logicals",
    )

    steane = get_code("steane")
    draw_tableau(
        steane,
        ["$S_{X1}$", "$S_{X2}$", "$S_{X3}$", "$S_{Z1}$", "$S_{Z2}$", "$S_{Z3}$",
         "$\\bar{X}$", "$\\bar{Z}$"],
        steane.stabilizers + [steane.Xbar, steane.Zbar],
        str(FIGURES_DIR / "tableau_steane.png"),
        "Steane [[7,1,3]] code: stabilizers and logicals",
    )
    
    print("Stabilizer tableau figures written to figures/")

if __name__ == "__main__":
    main()
