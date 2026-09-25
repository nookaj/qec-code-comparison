"""
Full quantum circuit simulation (Qiskit) implementation of the QEC pipeline: encode ->
inject noise as explicit Pauli gates -> measure every stabilizer with a
ancilla-based circuit -> classically-conditioned correction ->
fidelity. Also draws circuit diagrams and runs an extended validation
sweep against the stabilizer Monte Carlo (simulate.py).
  - The exact logical codeword (|0>_L or |+>_L) is built via
    qiskit.quantum_info.StabilizerState.from_stabilizer_list(), which
    constructs a stabilizer state directly from a list of Pauli-string
    generators
  - Fidelities are computed via qiskit.quantum_info.state_fidelity and
    Statevector.evolve(Pauli(...))
"""

import os
import csv
from pathlib import Path
import time
import numpy as np

from codes import get_code, syndrome
from noise import sample_error, sample_error_batch
from simulate import run_trials_vectorized

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, StabilizerState, Pauli, partial_trace, state_fidelity

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

"""
Pauli-string conversion: this project's (x,z) vectors -> Qiskit
label strings. Qiskit writes Pauli labels with qubit 0 as the rightmost
character, while codes.py's pauli_from_string() reads a string left-to-right
as qubit 0,1,2,..., so this reverses the order. 
"""
def xz_to_qiskit_label(x, z):
    n = len(x)
    chars = []
    for i in range(n - 1, -1, -1):
        if x[i] and z[i]:
            chars.append("Y")
        elif x[i]:
            chars.append("X")
        elif z[i]:
            chars.append("Z")
        else:
            chars.append("I")
    return "".join(chars)

def build_correction_table_qiskit_order(code):
    """Re-key codes.py's syndrome->correction table to match Qiskit's
    classical-register integer convention: creg holds one bit per
    stabilizer (bit i = measurement of stabilizer i), and Qiskit's
    int(creg) == sum(bit_i * 2**i)."""
    k = len(code.stabilizers)
    table = {}
    for bits_tuple, (cx, cz) in code.table.items():
        idx = sum(int(b) << i for i, b in enumerate(bits_tuple))
        table[idx] = (cx, cz)
    assert len(table) == 2 ** k, "table should cover every syndrome"
    return table

def build_codeword_circuit(code, logical="zero"):
    """Build a QuantumCircuit that prepares the exact logical |0>_L or
    |+>_L codeword,: specify n independent stabilizer generators
    (the code's n-k generators plus whichever
    logical operator fixes the remaining degree of freedom), let
    StabilizerState figure out a state satisfying all of them, then pull
    a preparation circuit out of it."""
    labels = [xz_to_qiskit_label(xs, zs) for xs, zs in code.stabilizers]
    if logical == "zero":
        labels.append(xz_to_qiskit_label(*code.Zbar))   # Z_bar = +1  ->  |0>_L
    elif logical == "plus":
        labels.append(xz_to_qiskit_label(*code.Xbar))   # X_bar = +1  ->  |+>_L
    else:
        raise ValueError(logical)

    stab_state = StabilizerState.from_stabilizer_list(labels)
    try:
        prep_circuit = stab_state.clifford.to_circuit()
    except AttributeError as e:
        raise AttributeError(
            "StabilizerState has no '.clifford' attribute on this Qiskit "
            "version. Run `print(dir(stab_state))` to find the right way "
            "to get a Clifford/circuit out of it on your installed "
            "version, then adjust build_codeword_circuit() here."
        ) from e
    return prep_circuit

def verify_codeword_circuit(code, logical="zero", tol=1e-6):
    """Self-check: build the codeword circuit, then verify using
    Qiskit's Statevector.expectation_value that the resulting state
    really does satisfy every stabilizer generator and the intended
    logical operator."""
    prep = build_codeword_circuit(code, logical)
    sv = Statevector(prep)
    for xs, zs in code.stabilizers:
        label = xz_to_qiskit_label(xs, zs)
        ev = sv.expectation_value(Pauli(label))
        assert abs(ev - 1.0) < tol, f"stabilizer {label} not satisfied (eigenvalue {ev})"
    logical_op = code.Zbar if logical == "zero" else code.Xbar
    logical_label = xz_to_qiskit_label(*logical_op)
    ev = sv.expectation_value(Pauli(logical_label))
    assert abs(ev - 1.0) < tol, f"logical operator {logical_label} not satisfied (eigenvalue {ev})"
    return True

def build_trial_circuit(code, xe, ze, corr_table, plus_prep_circuit):
    """Build the full single-shot circuit: encode |+>_L (using the
    precomputed preparation circuit, built once per code, not rebuilt
    every trial), inject the given error, extract every syndrome bit via
    its own ancilla, and apply a classically-conditioned correction for
    every possible syndrome value."""
    n, k = code.n, len(code.stabilizers)

    data = QuantumRegister(n, "data")
    anc = QuantumRegister(k, "anc")
    creg = ClassicalRegister(k, "syndrome")
    qc = QuantumCircuit(data, anc, creg)

    # 1) encode |+>_L
    qc.compose(plus_prep_circuit, qubits=data, inplace=True)
    qc.barrier()

    # 2) inject noise as explicit Pauli gates on the data qubits
    for i in range(n):
        if xe[i] and ze[i]:
            qc.y(data[i])
        elif xe[i]:
            qc.x(data[i])
        elif ze[i]:
            qc.z(data[i])
    qc.barrier()

    # 3) syndrome extraction: standard ancilla-based stabilizer measurement,
    # one fresh ancilla per generator: H -> controlled-Pauli -> H -> measure
    for si, (xs, zs) in enumerate(code.stabilizers):
        qc.h(anc[si])
        for qi in range(n):
            if xs[qi] and zs[qi]:
                qc.cy(anc[si], data[qi])
            elif xs[qi]:
                qc.cx(anc[si], data[qi])
            elif zs[qi]:
                qc.cz(anc[si], data[qi])
        qc.h(anc[si])
        qc.measure(anc[si], creg[si])
    qc.barrier()

    # 4) classically-conditioned correction, one branch per reachable
    #   syndrome value
    for synd_int, (cx, cz) in corr_table.items():
        if synd_int == 0:
            continue  # no correction needed
        with qc.if_test((creg, synd_int)):
            for qi in range(n):
                if cx[qi] and cz[qi]:
                    qc.z(data[qi])
                    qc.x(data[qi])
                elif cx[qi]:
                    qc.x(data[qi])
                elif cz[qi]:
                    qc.z(data[qi])

    qc.save_statevector(label="final")
    return qc

def draw_full_pipeline_circuit(code_key="repetition", save_path=str(FIGURES_DIR / "full_pipeline_repetition.png")):
    """Draw the complete pipeline circuit for a small code, with one
    illustrative error inserted. Only sensible for the repetition code
    (5 total qubits) -- use draw_stabilizer_subcircuit() for the larger
    codes, whose full circuits are too large to read as one diagram."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    code = get_code(code_key)
    corr_table = build_correction_table_qiskit_order(code)
    plus_prep = build_codeword_circuit(code, "plus")
    xe = np.zeros(code.n, dtype=int)
    ze = np.zeros(code.n, dtype=int)
    xe[min(1, code.n - 1)] = 1  # illustrative: X on qubit 1
    qc = build_trial_circuit(code, xe, ze, corr_table, plus_prep)
    fig = qc.draw(output="mpl", fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {save_path}")


def draw_stabilizer_subcircuit(code_key="five_qubit", stab_index=0,
                                save_path=str(FIGURES_DIR / "stabilizer_subcircuit.png")):
    """Draw just ONE stabilizer's syndrome-extraction circuit in isolation
    -- the general pattern repeated once per stabilizer generator."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    code = get_code(code_key)
    n = code.n
    xs, zs = code.stabilizers[stab_index]

    data = QuantumRegister(n, "data")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(1, "syndrome")
    qc = QuantumCircuit(data, anc, creg)
    qc.h(anc[0])
    for qi in range(n):
        if xs[qi] and zs[qi]:
            qc.cy(anc[0], data[qi])
        elif xs[qi]:
            qc.cx(anc[0], data[qi])
        elif zs[qi]:
            qc.cz(anc[0], data[qi])
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])

    labels = []
    for x, z in zip(xs, zs):
        if x and z:
            labels.append("Y")
        elif x:
            labels.append("X")
        elif z:
            labels.append("Z")
        else:
            labels.append("I")
    fig = qc.draw(output="mpl", fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {save_path}  (stabilizer = {''.join(labels)})")

def run_one_trial(code, noise_type, p, rng, backend, corr_table, plus_prep_circuit, zero_sv, plus_sv):
    n, k = code.n, len(code.stabilizers)
    xe, ze = sample_error(n, noise_type, p, rng)

    qc = build_trial_circuit(code, xe, ze, corr_table, plus_prep_circuit)
    tqc = transpile(qc, backend, optimization_level=0)
    result = backend.run(tqc, shots=1, memory=True).result()
    creg_str = result.get_memory(tqc)[0]
    synd_int = int(creg_str, 2)
    full_sv = Statevector(result.data(tqc)["final"])

    # trace out the ancillas -> reduced state on the data qubits only
    reduced_dm = partial_trace(full_sv, list(range(n, n + k)))
    fid_plus = state_fidelity(reduced_dm, plus_sv)

    # cross-check against |0>_L: same (xe, ze) and the same measured
    # correction, applied via Qiskit's Statevector.evolve(Pauli(...))
    # instead of a full second circuit run.
    cx, cz = corr_table[synd_int]
    error_label = xz_to_qiskit_label(xe, ze)
    corr_label = xz_to_qiskit_label(cx, cz)
    zero_after = zero_sv.evolve(Pauli(error_label)).evolve(Pauli(corr_label))
    fid_zero = state_fidelity(zero_sv, zero_after)

    return (fid_plus > 0.999) and (fid_zero > 0.999)

def circuit_level_P_L_qiskit(code_key, noise_type, p, n_trials, rng, backend):
    code = get_code(code_key)
    corr_table = build_correction_table_qiskit_order(code)
    # built once per code (not per trial)
    plus_prep_circuit = build_codeword_circuit(code, "plus")
    zero_sv = Statevector(build_codeword_circuit(code, "zero"))
    plus_sv = Statevector(plus_prep_circuit)

    n_fail = 0
    for _ in range(n_trials):
        ok = run_one_trial(code, noise_type, p, rng, backend, corr_table,
                            plus_prep_circuit, zero_sv, plus_sv)
        if not ok:
            n_fail += 1
    return n_fail / n_trials

if __name__ == "__main__":
    print("Drawing circuit diagrams...")
    draw_full_pipeline_circuit("repetition")
    draw_stabilizer_subcircuit("five_qubit", stab_index=0)
    print()

    # --- extended validation sweep -------------------------------------
    # Tune n_trials to available time. Steane will
    # dominate the runtime, so lower this if it's too slow.
    n_trials = 1500
    p_values = [0.02, 0.05, 0.1, 0.2]
    noise_models = ["bitflip", "phaseflip", "depolarizing"]
    code_keys = ["repetition", "five_qubit", "steane"]
    mc_trials = 2_000_000  # cheap and exact

    rng = np.random.default_rng(7)
    backend = AerSimulator(method="statevector")

    rows = []
    t0 = time.time()
    print(f"{'code':22s} {'noise':13s} {'p':>6s} {'P_L (qiskit)':>14s} {'P_L (fast MC)':>14s}")
    for code_key in code_keys:
        code = get_code(code_key)
        for noise in noise_models:
            for p in p_values:
                t_config = time.time()
                pl_qiskit = circuit_level_P_L_qiskit(code_key, noise, p, n_trials, rng, backend)
                pl_fast = run_trials_vectorized(code, noise, p, mc_trials, rng, sample_error_batch)
                rows.append((code.name, noise, p, pl_qiskit, pl_fast))
                print(f"{code.name:22s} {noise:13s} {p:6.2f} {pl_qiskit:14.4f} {pl_fast:14.4f}"
                      f"   ({time.time()-t_config:.1f}s)")
    print(f"\nDone in {time.time()-t0:.1f}s total")

    with open(RESULTS_DIR / "qiskit_validation.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code_name", "noise", "p", "P_L_qiskit", "P_L_fast_mc"])
            w.writerows(rows)
    print("Wrote results/qiskit_validation.csv")
    print("Next: run make_qiskit_validation_plot.py (no qiskit needed) to build the comparison figure.")
