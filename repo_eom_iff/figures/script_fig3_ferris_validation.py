"""
Figure 3 — Experimental Validation: Ferris (1996) Clay Catalysis Synergy
=========================================================================
Reproduces the Prediction II validation against Ferris et al. (1996) data:
  - Superlinearity factor: L_both / L_max_add = 50/12 ≈ 4.2
  - Synergy fraction: ΔΦ_I(synergy) / ΔΦ_I(total) ≈ 83%
  - Result is EXACTLY independent of γ (SI Table S4)

Four experimental conditions from Ferris et al. (1996):
  1. Bulk aqueous:        L_max ≈ 4  monomers
  2. Catalysis only:      L_max ≈ 10 monomers
  3. Confinement only:    L_max ≈ 6  monomers
  4. Cat. + confinement:  L_max ≈ 50 monomers (observed superlinearity)

EOM-IFF prediction: ΔΦ_I(cat+conf) > ΔΦ_I(cat) + ΔΦ_I(conf)
Additive prediction: L_max^add = 4 + (10-4) + (6-4) = 12

Usage:
  python figures/script_fig3_ferris_validation.py

Output:
  figures/fig3_experimental_validation.png

Reference:
  Ferris, J.P. et al. (1996). Synthesis of long prebiotic oligomers
  on mineral surfaces. Nature, 381, 59-61.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

DARK  = '#0d1117'; PANEL = '#0d1b2a'; GRID_C = '#1e2a3a'; TEXT = '#e0e0e0'
C_Y = '#ffd54f'; C_B = '#4fc3f7'; C_G = '#66bb6a'; C_R = '#ef5350'; C_OR = '#ffa726'

# ── Ferris (1996) data ────────────────────────────────────────────────────
conditions   = ['Bulk\naqueous', 'Catalysis\nonly', 'Confinement\nonly', 'Cat. +\nConfinement']
L_max_obs    = np.array([4, 10, 6, 50])       # observed max chain length
L_max_err    = np.array([1, 2,  1.5, 8])       # approximate uncertainty

# EOM-IFF Prediction II
L_base = L_max_obs[0]
dL_cat  = L_max_obs[1] - L_base   # = 6
dL_conf = L_max_obs[2] - L_base   # = 2
L_add   = L_base + dL_cat + dL_conf  # additive prediction = 12
L_obs   = L_max_obs[3]               # observed = 50

superlinearity = L_obs / L_add
print(f"Additive prediction: {L_add}")
print(f"Observed L_max:      {L_obs}")
print(f"Superlinearity factor: {superlinearity:.2f}  (paper: 4.2)")

# Synergy fraction
gamma   = 0.15  # kBT / monomer (best estimate from SantaLucia 1998)
dPhi_cat  = gamma * dL_cat
dPhi_conf = gamma * dL_conf
dPhi_both = gamma * (L_obs - L_base)
dPhi_syn  = dPhi_both - dPhi_cat - dPhi_conf
synergy_frac = dPhi_syn / dPhi_both
print(f"Synergy fraction:    {synergy_frac:.3f}  (paper: 0.83)")

# ── Panel A: Bar chart of conditions ─────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40,
                        left=0.07, right=0.97, top=0.88, bottom=0.14)

def sax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.4, axis='y')

ax1 = fig.add_subplot(gs[0]); sax(ax1)
colors_bar = [C_B, C_G, C_OR, C_R]
x = np.arange(4)
bars = ax1.bar(x, L_max_obs, yerr=L_max_err, color=colors_bar,
               alpha=0.85, edgecolor=DARK, lw=0.8, width=0.55,
               capsize=5, error_kw={'color': TEXT, 'lw': 1.5})

# Additive prediction line
ax1.axhline(L_add, color=C_Y, ls='--', lw=2, alpha=0.8,
            label=f'Additive prediction = {L_add}')
ax1.fill_between([-0.5, 3.5], L_add, L_max_obs[3],
                 alpha=0.08, color=C_R)

# Annotations
ax1.annotate('', xy=(3, L_max_obs[3] - 2), xytext=(3, L_add + 1),
             arrowprops=dict(arrowstyle='<->', color=C_R, lw=2))
ax1.text(3.25, (L_max_obs[3] + L_add) / 2,
         f'Synergy\n×{superlinearity:.1f}',
         color=C_R, fontsize=9, va='center', fontweight='bold')

ax1.text(3, L_max_obs[3] + 3, f'L = {int(L_obs)}', color=TEXT,
         ha='center', fontsize=10, fontweight='bold')
ax1.text(2.5, L_add + 2, f'Additive\n= {L_add}', color=C_Y,
         ha='center', fontsize=8.5)

ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=9.5)
ax1.set_ylabel('Maximum oligomer length (monomers)', fontsize=10)
ax1.set_title('Panel A — Observed vs Additive Prediction\nFerris et al. (1996)',
              fontsize=10, pad=5)
ax1.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT,
           facecolor=PANEL, edgecolor=GRID_C)
ax1.set_ylim(0, 65)

# ── Panel B: ΔΦ_I decomposition ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[1]); sax(ax2)

# Stacked bar: catalysis + confinement + synergy
components   = [dPhi_cat, dPhi_conf, dPhi_syn]
comp_labels  = [f'Catalysis\n({dPhi_cat/dPhi_both*100:.0f}%)',
                f'Confinement\n({dPhi_conf/dPhi_both*100:.0f}%)',
                f'Synergy\n({synergy_frac*100:.0f}%)']
comp_colors  = [C_G, C_OR, C_R]

bottom = 0
for comp, label, color in zip(components, comp_labels, comp_colors):
    ax2.bar(0, comp, bottom=bottom, color=color, alpha=0.85,
            edgecolor=DARK, lw=0.8, width=0.4, label=label)
    mid = bottom + comp/2
    ax2.text(0.25, mid, f'{comp/dPhi_both*100:.0f}%',
             color=TEXT, va='center', fontsize=10, fontweight='bold')
    bottom += comp

ax2.set_xlim(-0.5, 1.5)
ax2.set_xticks([0])
ax2.set_xticklabels(['Cat. + Conf.'], fontsize=10)
ax2.set_ylabel(r'$\Delta\Phi_I$ contribution ($k_BT$, $\gamma=0.15$)', fontsize=10)
ax2.set_title(f'Panel B — Quasi-Potential Decomposition\n'
              f'Synergy = {synergy_frac*100:.0f}% of total $\\Delta\\Phi_I$',
              fontsize=10, pad=5)
ax2.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT,
           facecolor=PANEL, edgecolor=GRID_C, loc='upper right')

# Note: independent of gamma
ax2.text(0, -dPhi_both*0.12,
         f'Synergy fraction independent of $\\gamma$\n(SI Table S4, verified)',
         color='#90a4ae', fontsize=8, ha='center')

fig.suptitle(
    'Figure 3 — Prediction II Validation: Superlinear Catalysis–Confinement Synergy',
    color=TEXT, fontsize=12, y=0.975, fontweight='bold')

plt.savefig('figures/fig3_experimental_validation.png', dpi=160,
            bbox_inches='tight', facecolor=DARK)
print(f"Saved figures/fig3_experimental_validation.png")
print(f"Superlinearity: {superlinearity:.2f}x  |  Synergy: {synergy_frac*100:.0f}%")
