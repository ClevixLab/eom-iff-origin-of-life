"""
Script 01 — Five-State Amino-Acid Markov Chain
===============================================
Computes the verified stationary distribution p*(x) and information
quasi-potential Phi_I(x) = -ln p*(x) for the five-state prebiotic network.

Reproduces:
  - Table I in the main paper (p* and Phi_I values)
  - SI Section S1 (rate matrix K, numerical note on solver)

Key result:
  p*(Asp) = 0.3254  ->  deepest attractor
  Ranking: Asp > Ala ≈ Gly > Ser  (consistent with Murchison/Ryugu inventories)

IMPORTANT — solver note:
  Direct linear solvers fail on K^T (condition number ~10^16).
  We use scipy.linalg.eig (full eigendecomposition) and select the
  eigenvector for eigenvalue closest to zero.
  Residual ||Kp*||_inf < 10^-20 confirms accuracy.

Usage:
  python scripts/01_markov_chain_pstar.py

Dependencies: numpy>=1.20, scipy>=1.7
"""

import numpy as np
from scipy.linalg import eig

# ── Parameters (from paper Section II.D) ─────────────────────────────────
PHI   = 5e-6   # entropy flux (s^-1)
C     = np.array([1.0, 0.9, 0.7, 1.1])          # synthesis coefficients c_i
K_HYD = np.array([3.2, 2.8, 4.1, 2.1]) * 1e-6  # hydrolysis rates (s^-1)
K_IC  = 1e-7   # interconversion rate (s^-1), same for all amino-acid pairs
LABELS = ['Gly', 'Ala', 'Ser', 'Asp', 'Deg']
N = 5


def build_rate_matrix(phi=PHI, c=C, k_hyd=K_HYD, k_ic=K_IC):
    """
    Build the generator matrix K for the five-state network.

    Convention:
      K[i,j] = rate FROM state j TO state i   (i != j, non-negative)
      K[i,i] = -sum_{j!=i} K[j,i]            (diagonal, non-positive)
      => each COLUMN sums to zero (probability conservation)

    States: 0=Gly, 1=Ala, 2=Ser, 3=Asp, 4=Deg
    """
    K = np.zeros((N, N))
    # Synthesis: Deg(4) -> amino acid i
    for i in range(4):
        K[i, 4] = c[i] * phi
    # Hydrolysis: amino acid i -> Deg(4)
    for i in range(4):
        K[4, i] = k_hyd[i]
    # Interconversion among amino acids
    for i in range(4):
        for j in range(4):
            if i != j:
                K[j, i] += k_ic
    # Diagonal
    for i in range(N):
        K[i, i] = 0.0
        K[i, i] = -K[:, i].sum()
    return K


def stationary_distribution(K):
    """
    Compute p* via full eigendecomposition of K.

    Returns p* (normalised), the null eigenvalue, and the residual.
    """
    vals, vecs = eig(K)
    idx = np.argmin(np.abs(vals))
    p = np.real(vecs[:, idx])
    if p.sum() < 0:
        p = -p
    p = np.abs(p)
    p /= p.sum()
    residual = np.max(np.abs(K @ p))
    return p, vals[idx], residual


def main():
    K = build_rate_matrix()

    # Sanity check: column sums must be zero
    col_sums = K.sum(axis=0)
    assert np.allclose(col_sums, 0, atol=1e-20), \
        f"Column sums not zero: {col_sums}"

    p, lam0, residual = stationary_distribution(K)
    phi_I = -np.log(p)

    # ── Print results ──────────────────────────────────────────────────────
    print("=" * 62)
    print("  Five-State Markov Chain — Verified Stationary Distribution")
    print("=" * 62)
    print(f"\n  Entropy flux Phi = {PHI:.1e} s^-1")
    print(f"  Null eigenvalue  = {lam0:.4e}  (should be ~0)")
    print(f"  Residual ||Kp*|| = {residual:.2e}  (should be <1e-20)")
    print()

    print(f"  {'State':>6}  {'p*(x)':>10}  {'Phi_I(x)':>10}  {'Rank':>4}")
    print("  " + "-" * 38)
    ranks = np.argsort(phi_I) + 1
    for i, lab in enumerate(LABELS):
        marker = "  <- deepest" if i == np.argmin(phi_I) else ""
        print(f"  {lab:>6}  {p[i]:>10.6f}  {phi_I[i]:>10.6f}  "
              f"{ranks[i]:>4}{marker}")

    print(f"\n  Sum p* = {p.sum():.10f}  (should be 1.0000000000)")

    # ── Rate matrix printout ───────────────────────────────────────────────
    print("\n  Rate matrix K (units: 10^-6 s^-1):")
    print(f"  {'':>6}", end="")
    for lab in LABELS:
        print(f"  {lab:>8}", end="")
    print()
    for i, li in enumerate(LABELS):
        print(f"  {li:>6}", end="")
        for j in range(N):
            print(f"  {K[i,j]*1e6:>8.3f}", end="")
        print()

    # ── Verification against paper values (Table I) ────────────────────────
    print("\n  Verification against Table I (paper v7.4):")
    expected = {
        'Gly': (0.20759, 1.5722),
        'Ala': (0.21290, 1.5469),
        'Ser': (0.12204, 2.1034),
        'Asp': (0.32536, 1.1228),
        'Deg': (0.13211, 2.0241),
    }
    all_ok = True
    for lab, (p_exp, phi_exp) in expected.items():
        i = LABELS.index(lab)
        ok = abs(p[i] - p_exp) < 1e-4 and abs(phi_I[i] - phi_exp) < 1e-3
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {lab}: p*={p[i]:.5f} (paper {p_exp:.5f}),"
              f"  Phi_I={phi_I[i]:.4f} (paper {phi_exp:.4f})")

    print()
    if all_ok:
        print("  RESULT: All values match paper Table I exactly.")
    else:
        print("  WARNING: Mismatch detected — check scipy version.")
    return p, phi_I, K


if __name__ == "__main__":
    main()
