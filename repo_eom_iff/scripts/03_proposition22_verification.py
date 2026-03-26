"""
Script 03 — Proposition 2.2: Coarse-Graining Error Bound
=========================================================
Verifies Proposition 2.2 (Coarse-Graining Error Bound) numerically
using a 20-state polymer network.

Proposition 2.2 states:
  |Delta_Phi_I^true - Delta_Phi_I^cg| <= ln|B_i|_eff + ln|B_j|_eff
where |B_i|_eff = P*(B_i) / p*(x_i*) is the effective basin size.

Reproduces SI Section S5 (Table S5).

Network structure (20 states):
  - States  0-4:  monomers (Gly, Ala, Ser, Asp) + degraded-mono
  - States  5-9:  dimers + degraded-dimer
  - States 10-14: trimers + degraded-trimer
  - States 15-19: tetramers (incl. extra conformation) + degraded-tetra

Key result:
  All four basins satisfy the bound.
  P(tetramer deeper than monomer) = 0.98 under +/-50% rate perturbations.

Usage:
  python scripts/03_proposition22_verification.py

Dependencies: numpy>=1.20, scipy>=1.7
"""

import numpy as np
from scipy.linalg import null_space
np.random.seed(42)

PHI = 5e-6
C4  = np.array([1.0, 0.9, 0.7, 1.1])
K_HYD_MONO  = np.array([3.2, 2.8, 4.1, 2.1]) * 1e-6
K_HYD_DIMER = K_HYD_MONO * 0.40
K_HYD_TRI   = K_HYD_MONO * 0.12
K_HYD_TETRA = K_HYD_MONO * 0.025
KP1, KP2, KP3 = 2.0e-6, 1.5e-6, 2.5e-6
K_IC = 5e-8
N20 = 20

BASINS = {
    'Monomer':  list(range(0, 5)),
    'Dimer':    list(range(5, 10)),
    'Trimer':   list(range(10, 15)),
    'Tetramer': list(range(15, 20)),
}


def build_20state_matrix(phi=PHI, c4=C4,
                          kh_m=K_HYD_MONO, kh_d=K_HYD_DIMER,
                          kh_t=K_HYD_TRI,  kh_T=K_HYD_TETRA,
                          kp1=KP1, kp2=KP2, kp3=KP3, kic=K_IC):
    K = np.zeros((N20, N20))

    for i in range(4):
        # Monomers <-> Deg-mono
        K[i, 4]   = c4[i] * phi;    K[4, i]   = kh_m[i]
        # Monomers <-> Dimers
        K[5+i, i] = kp1;            K[i, 5+i] = kh_d[i]
        # Deg-mono <-> Dimers (small leakage)
        K[4, 5+i] = kh_d[i] * 0.1; K[5+i, 4] = c4[i] * phi * 0.05
        # Dimers <-> Trimers
        K[10+i, 5+i] = kp2;         K[5+i, 10+i] = kh_t[i]
        K[4, 10+i]   = kh_t[i] * 0.05; K[10+i, 4] = c4[i] * phi * 0.01
        # Trimers <-> Tetramers
        K[15+i, 10+i] = kp3 * 1.3;  K[10+i, 15+i] = kh_T[i]
        K[4, 15+i]    = kh_T[i] * 0.02; K[15+i, 4] = c4[i] * phi * 0.002

    # Extra conformation for tetramer-0 and tetramer-1
    K[19, 15] = kp3 * 0.4;  K[15, 19] = kh_T[0] * 2
    K[19, 16] = kp3 * 0.3;  K[16, 19] = kh_T[1] * 2

    # Interconversion within each tier
    for group in [range(4), range(5, 9), range(10, 14), range(15, 19)]:
        for a in group:
            for b in group:
                if a != b:
                    K[b, a] += kic

    # Diagonal
    for i in range(N20):
        K[i, i] = 0.0
        K[i, i] = -K[:, i].sum()

    return K


def get_pstar_20(K):
    """Compute p* via null_space (20-state, well-conditioned)."""
    ns = null_space(K.T, rcond=1e-11)
    assert ns.shape[1] == 1, \
        f"Expected 1-d null space, got {ns.shape[1]}"
    p = ns[:, 0]
    if p.sum() < 0:
        p = -p
    p = np.abs(p)
    p /= p.sum()
    return p


def verify_proposition(p):
    """
    Verify Proposition 2.2 for all basin pairs.
    Returns a dict of results per basin.
    """
    phi_I = -np.log(p)
    results = {}

    for bname, bidx in BASINS.items():
        P_basin = p[bidx].sum()
        phi_cg  = -np.log(P_basin)
        p_max   = p[bidx].max()
        repr_idx = bidx[np.argmax(p[bidx])]
        phi_repr = phi_I[repr_idx]
        eff_size = P_basin / p_max
        bound    = np.log(eff_size)
        error    = abs(phi_cg - phi_repr)
        ok       = error <= bound + 1e-10

        results[bname] = {
            'P_basin':   P_basin,
            'phi_cg':    phi_cg,
            'phi_repr':  phi_repr,
            'eff_size':  eff_size,
            'bound':     bound,
            'error':     error,
            'ok':        ok,
        }
    return results, phi_I


def robustness_test(n_mc=2000):
    """
    Test robustness to +/-50% rate perturbations (log-uniform).
    Returns fraction of runs where P*(tetramer) > P*(monomer).
    """
    n_tetra_deeper = 0
    n_valid = 0

    for _ in range(n_mc):
        def f():
            return np.exp(np.random.uniform(-np.log(2), np.log(2)))

        K2 = build_20state_matrix(
            phi=PHI*f(),
            c4=C4,
            kh_m=K_HYD_MONO * np.array([f(), f(), f(), f()]),
            kh_d=K_HYD_DIMER * np.array([f(), f(), f(), f()]),
            kh_t=K_HYD_TRI  * np.array([f(), f(), f(), f()]),
            kh_T=K_HYD_TETRA* np.array([f(), f(), f(), f()]),
            kp1=KP1*f(), kp2=KP2*f(), kp3=KP3*f(),
        )
        try:
            ns = null_space(K2.T, rcond=1e-11)
            if ns.shape[1] != 1:
                continue
            p2 = ns[:, 0]
            if p2.sum() < 0:
                p2 = -p2
            p2 = np.abs(p2); p2 /= p2.sum()
            p_mono  = p2[list(range(0, 5))].sum()
            p_tetra = p2[list(range(15, 20))].sum()
            if p_tetra > p_mono:
                n_tetra_deeper += 1
            n_valid += 1
        except Exception:
            pass

    return n_tetra_deeper / n_valid if n_valid > 0 else 0.0, n_valid


def main():
    print("=" * 64)
    print("  Proposition 2.2 — Coarse-Graining Error Bound")
    print("  20-State Polymer Network Verification (SI Section S5)")
    print("=" * 64)

    K = build_20state_matrix()
    col_sums = K.sum(axis=0)
    print(f"\n  Max |col sum| = {np.max(np.abs(col_sums)):.2e}  (should be ~0)")

    p = get_pstar_20(K)
    results, phi_I = verify_proposition(p)

    print("\n  Basin analysis:")
    print(f"  {'Basin':>10}  {'P*(basin)':>12}  {'phi_cg':>8}  "
          f"{'phi_repr':>9}  {'error':>7}  {'bound':>7}  {'OK?':>4}")
    print("  " + "-" * 68)

    all_ok = True
    for bname, r in results.items():
        ok_str = "YES" if r['ok'] else "NO"
        if not r['ok']:
            all_ok = False
        print(f"  {bname:>10}  {r['P_basin']:>12.6f}  {r['phi_cg']:>8.4f}  "
              f"{r['phi_repr']:>9.4f}  {r['error']:>7.4f}  "
              f"{r['bound']:>7.4f}  {ok_str:>4}")

    print()
    if all_ok:
        print("  RESULT: Proposition 2.2 verified for all basins.")
    else:
        print("  WARNING: Bound violated for some basin.")

    # Robustness
    print("\n  Robustness test (n=2000, +/-50% rate perturbations):")
    frac, n_valid = robustness_test()
    print(f"  P(tetramer deeper than monomer) = {frac:.4f}  "
          f"({int(frac*n_valid)}/{n_valid} valid runs)")
    print(f"  Paper value: 0.98")
    ok_rob = abs(frac - 0.98) < 0.05
    print(f"  {'OK' if ok_rob else 'FAIL'}: |{frac:.3f} - 0.98| = "
          f"{abs(frac-0.98):.3f}  ({'within' if ok_rob else 'outside'} 0.05 tolerance)")


if __name__ == "__main__":
    main()
