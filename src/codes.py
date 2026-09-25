"""
Defines the three quantum error-correcting codes used in the project:
- 3-qubit bit-flip repetition code [[3,1,1]]
- 5-qubit perfect code [[5,1,3]]
- Steane code [[7,1,3]] 

Every Pauli operator (stabilizer generator, logical operator, physical error)
is represented in the binary form  P = (x, z),  x, z in {0,1}^n,
meaning P = i^(x.z) * prod_k X_k^{x_k} Z_k^{z_k}   (phases are irrelevant here,
we only ever need commutation relations and group membership).

Two Paulis (x1,z1), (x2,z2) commute  <=>  (x1.z2 + z1.x2) mod 2 == 0.

For each code we build:
  - stabilizer generators (list of (x,z) vectors)
  - logical X-bar, Z-bar representatives
  - a syndrome -> correction lookup table, built by enumerating all weight-1
    (single physical qubit) Pauli errors that are consistent with the code's
    error-correction claims (stabilizer-based decoding for the 5-qubit code,
    classical-Hamming lookup-table decoding for the Steane code, majority-vote
    lookup-table decoding for the repetition code).
"""

import numpy as np
import itertools

def commute(x1, z1, x2, z2):
    """Return True if the two Paulis (given in symplectic form) commute."""
    return int((np.dot(x1, z2) + np.dot(z1, x2)) % 2) == 0

def syndrome(xe, ze, stabilizers):
    """Compute the syndrome (list of 0/1 bits) of error (xe,ze) against a
    list of stabilizer generators [(xs,zs), ...]. Bit = 1 means the error
    anti-commutes with that stabilizer (i.e. it is detected)."""
    s = []
    for (xs, zs) in stabilizers:
        s.append(0 if commute(xe, ze, xs, zs) else 1)
    return tuple(s)

def pauli_from_string(pstr):
    """Convert a Pauli string like 'XZZXI' into symplectic (x,z) vectors."""
    x = np.zeros(len(pstr), dtype=int)
    z = np.zeros(len(pstr), dtype=int)
    for i, c in enumerate(pstr):
        if c == 'X':
            x[i] = 1
        elif c == 'Z':
            z[i] = 1
        elif c == 'Y':
            x[i] = 1
            z[i] = 1
        elif c == 'I':
            pass
        else:
            raise ValueError(f"bad Pauli char {c}")
    return x, z

def xor_vecs(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])

def weight1_paulis(n):
    """Yield (label, x, z) for every single-qubit X, Y, Z error on n qubits,
    used to build 'perfect' code style syndrome lookup tables."""
    out = []
    for q in range(n):
        for label, (xv, zv) in [('X', (1, 0)), ('Y', (1, 1)), ('Z', (0, 1))]:
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int)
            x[q] = xv
            z[q] = zv
            out.append((f"{label}{q}", x, z))
    return out

# ---------------------------------------------------------------------------
# 1) 3-qubit bit-flip repetition code [[3,1,1]]
# ---------------------------------------------------------------------------
class RepetitionCode:
    name = "3-qubit repetition code"
    n = 3

    def __init__(self):
        self.stabilizers = [pauli_from_string("ZZI"), pauli_from_string("IZZ")]
        self.Xbar = pauli_from_string("XXX")
        self.Zbar = pauli_from_string("ZZZ")
        # lookup table built from the 3 weight-1 X errors only: this code's
        # stabilizers are purely Z-type, so it is *blind* to Z errors by
        # construction (a deliberate, important limitation we study).
        self.table = {(0, 0): (np.zeros(3, dtype=int), np.zeros(3, dtype=int))}
        for q in range(3):
            x = np.zeros(3, dtype=int)
            x[q] = 1
            z = np.zeros(3, dtype=int)
            s = syndrome(x, z, self.stabilizers)
            self.table[s] = (x, z)

    def decode(self, xe, ze):
        s = syndrome(xe, ze, self.stabilizers)
        return self.table.get(s, (np.zeros(3, dtype=int), np.zeros(3, dtype=int)))

# ---------------------------------------------------------------------------
# 2) 5-qubit perfect code [[5,1,3]]
# ---------------------------------------------------------------------------
class FiveQubitCode:
    name = "5-qubit perfect code"
    n = 5

    def __init__(self):
        gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        self.stabilizers = [pauli_from_string(g) for g in gens]
        self.Xbar = pauli_from_string("XXXXX")
        self.Zbar = pauli_from_string("ZZZZZ")
        # Generic stabilizer-based decoding: enumerate all 15 weight-1 Pauli
        # errors, compute their 4-bit syndromes, and build the lookup table.
        # Because this is a "perfect" code, the map is a bijection onto the
        # 15 non-trivial syndromes (2^4 - 1 = 15).
        self.table = {(0, 0, 0, 0): (np.zeros(5, dtype=int), np.zeros(5, dtype=int))}
        for label, x, z in weight1_paulis(5):
            s = syndrome(x, z, self.stabilizers)
            assert s not in self.table, f"syndrome collision at {label}"
            self.table[s] = (x, z)
        assert len(self.table) == 16, "5-qubit code should be a perfect code (16 syndromes)"

    def decode(self, xe, ze):
        s = syndrome(xe, ze, self.stabilizers)
        return self.table.get(s, (np.zeros(5, dtype=int), np.zeros(5, dtype=int)))

# ---------------------------------------------------------------------------
# 3) Steane code [[7,1,3]] (CSS, from classical Hamming(7,4,3))
# ---------------------------------------------------------------------------
class SteaneCode:
    name = "Steane [[7,1,3]] code"
    n = 7

    def __init__(self):
        # Hamming(7,4) parity check matrix, columns = binary(1..7),
        # row_a = LSB row, row_b = middle bit row, row_c = MSB row.
        row_a = [1, 0, 1, 0, 1, 0, 1]  # bit0 of column index
        row_b = [0, 1, 1, 0, 0, 1, 1]  # bit1
        row_c = [0, 0, 0, 1, 1, 1, 1]  # bit2
        self.H = np.array([row_a, row_b, row_c])
        # X-type stabilizers detect Z errors, Z-type stabilizers detect X errors
        self.stab_X = [self._row_to_pauli(row, kind='X') for row in self.H]
        self.stab_Z = [self._row_to_pauli(row, kind='Z') for row in self.H]
        self.stabilizers = self.stab_X + self.stab_Z
        self.Xbar = pauli_from_string("X" * 7)
        self.Zbar = pauli_from_string("Z" * 7)
        # Full lookup table (classical Hamming-code decoding): the X-error
        # location (0=none,1..7=qubit) and Z-error location (0=none,1..7)
        # are each recovered independently and directly from a 3-bit
        # syndrome (stabilizer order = stab_X + stab_Z, matching self.stabilizers).
        self.table = {}
        for qx in range(8):      # 0 = no X error, else qubit index (1-indexed)
            for qz in range(8):  # 0 = no Z error, else qubit index (1-indexed)
                x = np.zeros(7, dtype=int)
                z = np.zeros(7, dtype=int)
                if qx:
                    x[qx - 1] = 1
                if qz:
                    z[qz - 1] = 1
                s = syndrome(x, z, self.stabilizers)  # stab_X then stab_Z
                assert s not in self.table, "Steane syndrome collision"
                self.table[s] = (x, z)
        assert len(self.table) == 64

    @staticmethod
    def _row_to_pauli(row, kind):
        n = len(row)
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        for i, b in enumerate(row):
            if b:
                if kind == 'X':
                    x[i] = 1
                else:
                    z[i] = 1
        return x, z

    def _hamming_lookup(self, synd_bits):
        """synd_bits = (s_a, s_b, s_c) -> qubit index 1..7, or 0 = no error.
        This is the classic Hamming lookup-table decoding: the syndrome,
        read as a binary number, directly IS the index of the faulty qubit."""
        s_a, s_b, s_c = synd_bits
        return s_a + 2 * s_b + 4 * s_c

    def decode(self, xe, ze):
        # syndrome from Z-stabilizers detects the X part of the error
        sZ = syndrome(xe, ze, self.stab_Z)   # sensitive to X errors
        sX = syndrome(xe, ze, self.stab_X)   # sensitive to Z errors
        qx = self._hamming_lookup(sZ)  # qubit with X error (1-indexed), 0=none
        qz = self._hamming_lookup(sX)  # qubit with Z error (1-indexed), 0=none
        cx = np.zeros(7, dtype=int)
        cz = np.zeros(7, dtype=int)
        if qx != 0:
            cx[qx - 1] = 1
        if qz != 0:
            cz[qz - 1] = 1
        return cx, cz

def bits_of(idx, k):
    """Integer -> tuple of k bits, MSB first."""
    return tuple((idx >> (k - 1 - i)) & 1 for i in range(k))

def add_vectorized_support(code):
    """Attach Sx, Sz check matrices and full syndrome -> correction lookup
    arrays (corr_x_table, corr_z_table) to a code instance, enabling a fully
    vectorized (numpy, no per-trial Python loop) Monte Carlo simulation."""
    k = len(code.stabilizers)
    n = code.n
    code.k = k
    code.Sx = np.array([xs for xs, zs in code.stabilizers])  # (k,n)
    code.Sz = np.array([zs for xs, zs in code.stabilizers])  # (k,n)
    corr_x_table = np.zeros((2 ** k, n), dtype=int)
    corr_z_table = np.zeros((2 ** k, n), dtype=int)
    for idx in range(2 ** k):
        bits = bits_of(idx, k)
        if bits in code.table:
            cx, cz = code.table[bits]
            corr_x_table[idx] = cx
            corr_z_table[idx] = cz
    code.corr_x_table = corr_x_table
    code.corr_z_table = corr_z_table
    code.weights = 2 ** np.arange(k - 1, -1, -1)

# ---------------------------------------------------------------------------
# Self-consistency checks (run at import time)
# ---------------------------------------------------------------------------
code_classes = {
    "repetition": RepetitionCode,
    "five_qubit": FiveQubitCode,
    "steane": SteaneCode,
}

def get_code(key):
    c = code_classes[key]()
    add_vectorized_support(c)
    return c

def _self_test():
    rng = np.random.default_rng(0)
    for key, Code in code_classes.items():
        c = Code()
        # 1) stabilizer generators must mutually commute
        for (xa, za), (xb, zb) in itertools.combinations(c.stabilizers, 2):
            assert commute(xa, za, xb, zb), f"{c.name}: stabilizers don't commute"
        # 2) logical Xbar, Zbar must commute with every stabilizer
        for (xs, zs) in c.stabilizers:
            assert commute(*c.Xbar, xs, zs), f"{c.name}: Xbar doesn't commute with stabilizer"
            assert commute(*c.Zbar, xs, zs), f"{c.name}: Zbar doesn't commute with stabilizer"
        # 3) logical Xbar and Zbar must anti-commute with each other
        assert not commute(*c.Xbar, *c.Zbar), f"{c.name}: Xbar,Zbar should anticommute"
        # 4) vectorized lookup table must agree with the scalar decode() on
        # random test errors
        add_vectorized_support(c)
        for _ in range(2000):
            xe = rng.integers(0, 2, size=c.n)
            ze = rng.integers(0, 2, size=c.n)
            cx, cz = c.decode(xe, ze)
            s = syndrome(xe, ze, c.stabilizers)
            idx = 0
            for b in s:
                idx = (idx << 1) | b
            assert np.array_equal(cx, c.corr_x_table[idx]), f"{c.name}: table mismatch (x)"
            assert np.array_equal(cz, c.corr_z_table[idx]), f"{c.name}: table mismatch (z)"
    print("All code self-consistency checks passed.")

if __name__ == "__main__":
    _self_test()
