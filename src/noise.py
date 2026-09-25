"""
Implements the three standard single-qubit Pauli noise channels applied
independently and identically to every physical qubit:
- bit-flip: X with prob p,   I otherwise
- phase-flip: Z with prob p,  I otherwise
- depolarizing:  X, Y or Z each with prob p/3,  I otherwise (total error prob = p)
"""

import numpy as np

def sample_error(n, noise_type, p, rng):
    """Return (xvec, zvec) error vectors for n qubits."""
    x = np.zeros(n, dtype=int)
    z = np.zeros(n, dtype=int)
    r = rng.random(n)
    if noise_type == "bitflip":
        hit = r < p
        x[hit] = 1
    elif noise_type == "phaseflip":
        hit = r < p
        z[hit] = 1
    elif noise_type == "depolarizing":
        hit = r < p
        # among hit qubits, choose X / Y / Z uniformly
        choice = rng.integers(0, 3, size=n)  # 0=X,1=Y,2=Z
        for q in range(n):
            if hit[q]:
                if choice[q] == 0:
                    x[q] = 1
                elif choice[q] == 1:
                    x[q] = 1
                    z[q] = 1
                else:
                    z[q] = 1
    return x, z

def sample_error_batch(n, noise_type, p, rng, M):
    """Vectorized version: return (X, Z) arrays of shape (M, n) representing
    M independent trials of the noise channel applied to n qubits."""
    X = np.zeros((M, n), dtype=int)
    Z = np.zeros((M, n), dtype=int)
    r = rng.random((M, n))
    hit = r < p
    if noise_type == "bitflip":
        X[hit] = 1
    elif noise_type == "phaseflip":
        Z[hit] = 1
    elif noise_type == "depolarizing":
        choice = rng.integers(0, 3, size=(M, n))  # 0=X,1=Y,2=Z
        is_x = hit & (choice == 0)
        is_y = hit & (choice == 1)
        is_z = hit & (choice == 2)
        X[is_x] = 1
        X[is_y] = 1
        Z[is_y] = 1
        Z[is_z] = 1
    return X, Z
