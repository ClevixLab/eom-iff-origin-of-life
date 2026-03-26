"""
SI Script S3 — Extended tau_cascade Table (SI Table S3)
=========================================================
Computes and prints the extended tau_cascade estimates for
15 planetary/astrophysical environments (SI Section S3).

For each environment:
  P = 1 - exp(-T / tau_cascade)
  P_lo = P at tau/10 (optimistic)
  P_hi = P at 10*tau (conservative)

Reproduces SI Table S3 exactly.

Usage:
  python SI/script_SI_S3_tau_extended.py

Dependencies: numpy
"""

import numpy as np

# ── 15-environment data (SI Table S3) ────────────────────────────────────
# Format: (name, T_yr, tau_yr, basis)
environments = [
    # ── High-probability ─────────────────────────────────────────────────
    ('Earth',
     4.0e9, 1e9,
     'Fossil record: first life ~3.5 Gyr; Hadean ocean available ~4.0 Gyr'),
    ('Proxima Cen b',
     3.5e9, 3e8,
     'M-dwarf long main-sequence; liquid water zone; τ scaled from Earth by flux'),
    ('TRAPPIST-1e',
     7.0e9, 5e8,
     'Old system (7.6 Gyr); habitable-zone candidate; τ from entropy flux estimate'),
    # ── Moderate-probability ─────────────────────────────────────────────
    ('Early Mars',
     5.0e8, 3e9,
     'Liquid water period ~0.5 Gyr; τ large due to intermittent flux'),
    ('Enceladus',
     1.0e9, 3e11,
     'Hydrothermal vents confirmed; small volume limits coupling; τ from volume scaling'),
    ('Europa',
     4.5e9, 1e11,
     'Subsurface ocean; radiation-processed organics; τ from ice-coverage penalty'),
    ('Titan',
     4.5e9, 1e12,
     'Organic chemistry; no liquid water; τ large (low-polarity solvent)'),
    # ── Low-probability ──────────────────────────────────────────────────
    ('Hot Jupiter HD189733b',
     1.0e9, 1e14,
     'High irradiation; no stable liquid phase; τ >> T_available'),
    ('Mars (present)',
     0.0,   1e13,
     'No liquid water; T_available ≈ 0 (surface sterilised)'),
    ('Venus',
     1.0e8, 1e12,
     'Early wet phase ~0.1 Gyr; runaway greenhouse; T_available short'),
    # ── Speculative ──────────────────────────────────────────────────────
    ('Ganymede (subsurface)',
     4.5e9, 5e11,
     'Deep ocean inferred; lower flux than Europa; τ scaled'),
    ('Ceres (ancient)',
     2.0e8, 2e10,
     'Briny water pockets; short wet phase; τ from low-gravity scaling'),
    ('K2-18b',
     3.0e9, 2e9,
     'Hycean candidate; H2-rich atmosphere; τ uncertain but plausible'),
    ('GJ 667Cc',
     5.0e9, 4e8,
     'Super-Earth in HZ; long-lived host; τ scaled from Earth'),
    ('Generic ISM cloud core',
     1.0e7, 1e15,
     'Pre-stellar chemistry; τ >> T due to low temperature and density'),
]

def compute_P(T, tau):
    if T <= 0 or tau <= 0:
        return 0.0
    return 1 - np.exp(-T / tau)

# ── Print table ───────────────────────────────────────────────────────────
print("=" * 90)
print("SI Table S3 — Extended tau_cascade Estimates (15 Environments)")
print("=" * 90)
print(f"{'Environment':25s}  {'T (yr)':>10s}  {'tau (yr)':>10s}  "
      f"{'P':>7s}  {'P_lo':>7s}  {'P_hi':>7s}")
print("-" * 90)

results = []
for name, T, tau, basis in environments:
    P     = compute_P(T, tau)
    P_lo  = compute_P(T, tau / 10)   # optimistic (shorter tau)
    P_hi  = compute_P(T, tau * 10)   # conservative (longer tau)
    results.append((name, T, tau, P, P_lo, P_hi, basis))

    def fmt(x):
        if x < 1e-4: return f'{x:.1e}'
        return f'{x:.4f}'

    print(f"  {name:23s}  {T:>10.2e}  {tau:>10.1e}  "
          f"{fmt(P):>7s}  {fmt(P_lo):>7s}  {fmt(P_hi):>7s}")

print("-" * 90)
print(f"\n  P = 1 - exp(-T/tau_cascade)  |  P_lo: tau/10 (optimistic)  |  P_hi: 10*tau (conservative)")

# ── Key statistics ────────────────────────────────────────────────────────
Ps = [r[3] for r in results if r[3] > 0]
print(f"\n=== Summary ===")
print(f"  Environments with P > 0.5:   {sum(1 for p in Ps if p > 0.5)}")
print(f"  Environments with P > 0.01:  {sum(1 for p in Ps if p > 0.01)}")
print(f"  Environments with P < 0.001: {sum(1 for p in Ps if p < 0.001)}")

# ── Verify paper-cited values ─────────────────────────────────────────────
print(f"\n=== Paper value verification ===")
checks = [
    ('Earth',        4.0e9, 1e9,   0.982),
    ('Enceladus',    1.0e9, 3e11,  0.004),
    ('Europa',       4.5e9, 1e11,  0.043),
]
for name, T, tau, P_paper in checks:
    P_calc = compute_P(T, tau)
    ok     = abs(P_calc - P_paper) < 0.01
    print(f"  {'OK' if ok else 'FAIL'}  {name:15s}: "
          f"P_calc={P_calc:.4f}  P_paper={P_paper:.3f}")
