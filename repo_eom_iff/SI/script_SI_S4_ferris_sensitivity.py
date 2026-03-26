"""
SI Script S4 — Ferris Synergy Sensitivity to gamma (SI Table S4)
=================================================================
Demonstrates that the superlinearity factor (4.2) and synergy
fraction (83%) are EXACTLY independent of the per-monomer depth
increment gamma in d(L) = d_0 + gamma*L.

This is a key result: Prediction II is robust to uncertainty in
the molecular energy parameter gamma.

Ferris et al. (1996) conditions:
  L_bulk = 4    (bulk aqueous)
  L_cat  = 10   (catalysis only)
  L_conf = 6    (confinement only)
  L_both = 50   (catalysis + confinement)

EOM-IFF prediction:
  Delta_Phi(cat+conf) > Delta_Phi(cat) + Delta_Phi(conf)
  Synergy fraction = Delta_Phi_syn / Delta_Phi_both = const(gamma)

Reproduces SI Table S4 exactly.

Usage:
  python SI/script_SI_S4_ferris_sensitivity.py

Dependencies: numpy
"""

import numpy as np

# ── Ferris (1996) oligomer lengths ────────────────────────────────────────
L_bulk = 4
L_cat  = 10
L_conf = 6
L_both = 50

# ── Gamma range (kBT / monomer) from SI ──────────────────────────────────
gamma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
# 0.15 is best estimate from SantaLucia (1998) RNA dissociation kinetics

print("=" * 80)
print("SI Table S4 — Ferris Synergy Sensitivity to gamma")
print("=" * 80)
print(f"{'gamma':>8s}  {'dPhi_cat':>10s}  {'dPhi_conf':>10s}  "
      f"{'dPhi_both':>10s}  {'dPhi_syn':>10s}  "
      f"{'Synergy%':>9s}  {'L_max/L_add':>11s}")
print("-" * 80)

all_synergy_fracs     = []
all_superlinearity    = []

for gamma in gamma_values:
    # Quasi-potential depth: Phi_I(L) = -ln p*(L) ~ gamma * L (linear model)
    dPhi_cat  = gamma * (L_cat  - L_bulk)   # = gamma * 6
    dPhi_conf = gamma * (L_conf - L_bulk)   # = gamma * 2
    dPhi_both = gamma * (L_both - L_bulk)   # = gamma * 46
    dPhi_add  = dPhi_cat + dPhi_conf        # additive prediction
    dPhi_syn  = dPhi_both - dPhi_add        # synergy term

    synergy_frac    = dPhi_syn / dPhi_both
    L_add           = L_bulk + (L_cat - L_bulk) + (L_conf - L_bulk)  # = 12
    superlinearity  = L_both / L_add          # = 50/12 ≈ 4.167

    all_synergy_fracs.append(synergy_frac)
    all_superlinearity.append(superlinearity)

    bold = " <-- best estimate" if abs(gamma - 0.15) < 0.001 else ""
    print(f"  {gamma:>6.2f}  {dPhi_cat:>10.4f}  {dPhi_conf:>10.4f}  "
          f"{dPhi_both:>10.4f}  {dPhi_syn:>10.4f}  "
          f"{synergy_frac*100:>8.1f}%  {superlinearity:>11.4f}{bold}")

print("-" * 80)
print(f"\n  L_add (additive prediction) = {L_add}  (= {L_bulk} + {L_cat-L_bulk} + {L_conf-L_bulk})")
print(f"  L_both (observed)            = {L_both}")
print(f"  Superlinearity factor        = {L_both}/{L_add} = {L_both/L_add:.4f}")

print(f"\n=== Key result: EXACT INDEPENDENCE FROM GAMMA ===")
print(f"  Synergy fraction range: {min(all_synergy_fracs)*100:.4f}% to "
      f"{max(all_synergy_fracs)*100:.4f}%")
print(f"  Superlinearity range:   {min(all_superlinearity):.6f} to "
      f"{max(all_superlinearity):.6f}")
print(f"  => Both are EXACTLY constant (gamma cancels algebraically)")

# ── Algebraic proof ───────────────────────────────────────────────────────
print(f"\n=== Algebraic explanation ===")
print(f"  dPhi_syn  = gamma*(L_both - L_bulk) - gamma*(L_cat-L_bulk) - gamma*(L_conf-L_bulk)")
print(f"            = gamma * [{L_both-L_bulk} - {L_cat-L_bulk} - {L_conf-L_bulk}]")
print(f"            = gamma * {(L_both-L_bulk) - (L_cat-L_bulk) - (L_conf-L_bulk)}")
print(f"  dPhi_both = gamma * {L_both - L_bulk}")
syn_exact = ((L_both-L_bulk) - (L_cat-L_bulk) - (L_conf-L_bulk)) / (L_both - L_bulk)
print(f"  Synergy   = {(L_both-L_bulk) - (L_cat-L_bulk) - (L_conf-L_bulk)}/{L_both-L_bulk} "
      f"= {syn_exact:.6f} = {syn_exact*100:.1f}% (exact, gamma cancels)")

# ── Verify paper values ───────────────────────────────────────────────────
print(f"\n=== Paper value verification ===")
checks = [
    ("Superlinearity = 50/12 ≈ 4.2", abs(L_both/L_add - 50/12) < 1e-10),
    ("Synergy = 83%",                abs(syn_exact - 0.8260869565) < 1e-6),
    ("L_add = 12",                   L_add == 12),
    ("Independence of gamma",        max(all_synergy_fracs)-min(all_synergy_fracs) < 1e-10),
]
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'}  {name}")
