"""
Monte Carlo engine. For a given code, noise model and physical error
probability p, runs n_trials repetitions of:
   1. sample a physical Pauli error E on the n physical qubits
   2. compute the syndrome and look up the correction C
   3. compute the residual R = E * C  (symplectic XOR)
   4. classify R:
        - commutes with both Xbar and Zbar  -> SUCCESS (R is a stabilizer
          element, or identity: acts trivially on the logical qubit)
        - anticommutes with Xbar and/or Zbar -> LOGICAL ERROR
This residual-commutation test is the general, code-agnostic way to detect
a logical error for any stabilizer code, and is exact (no approximation).
"""

import numpy as np
from codes import commute, xor_vecs

def run_trials(code, noise_type, p, n_trials, rng, sample_error):
    n_fail = 0
    for _ in range(n_trials):
        xe, ze = sample_error(code.n, noise_type, p, rng)
        cx, cz = code.decode(xe, ze)
        rx, rz = xor_vecs((xe, ze), (cx, cz))
        ok_x = commute(rx, rz, *code.Xbar)
        ok_z = commute(rx, rz, *code.Zbar)
        if not (ok_x and ok_z):
            n_fail += 1
    return n_fail / n_trials

def run_trials_vectorized(code, noise_type, p, n_trials, rng, sample_error_batch):
    """Fully vectorized Monte Carlo (numpy array ops only, no per-trial
    Python loop): scales to hundreds of thousands of trials quickly."""
    X, Z = sample_error_batch(code.n, noise_type, p, rng, n_trials)  # (M,n)
    synd = (X @ code.Sz.T + Z @ code.Sx.T) % 2          # (M,k)
    idx = synd @ code.weights                            # (M,)
    Cx = code.corr_x_table[idx]                           # (M,n)
    Cz = code.corr_z_table[idx]                           # (M,n)
    Rx = X ^ Cx
    Rz = Z ^ Cz
    Xbar_x, Xbar_z = code.Xbar   # Xbar is pure-X: Xbar_z is all zero
    Zbar_x, Zbar_z = code.Zbar   # Zbar is pure-Z: Zbar_x is all zero
    ok_x = ((Rz @ Xbar_x) % 2) == 0   # residual commutes with Xbar
    ok_z = ((Rx @ Zbar_z) % 2) == 0   # residual commutes with Zbar
    fail = ~(ok_x & ok_z)
    return fail.mean()

def unencoded_logical_error_rate(p):
    """A single unprotected physical qubit: by definition any non-identity
    Pauli error IS a logical error, and P(non-identity) = p for all three
    noise models as we've defined them (bit-flip, phase-flip, depolarizing
    all inject a nontrivial Pauli with total probability p per qubit)."""
    return p
