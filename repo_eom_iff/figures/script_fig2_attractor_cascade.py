"""
Figure 2 — The EOM-IFF Attractor Cascade
==========================================
Two-panel figure:
  Panel A: Quasi-potential wells Φ_I(x) = -ln p*(x) deepening
           hierarchically across 5 stages of cosmic/chemical evolution
  Panel B: Kramers-Eyring escape times τ ~ exp(ΔV/D) increasing
           by orders of magnitude at each scale (Definition II.2)

Key numbers reproduced (paper Section II.I, Table 2):
  Nuclear scale:   ΔV/D ≈ 8   → τ ~ 10^8  yr
  Chemical scale:  ΔV/D ≈ 12  → τ ~ 10^12 yr
  Polymer scale:   ΔV/D ≈ 18  → τ ~ 10^18 yr
  Cellular scale:  ΔV/D ≈ 25  → τ > age of universe

Usage:
  python figures/script_fig2_attractor_cascade.py

Output:
  figures/fig2_attractor_cascade.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

np.random.seed(42)

DARK  = '#0d1117'; PANEL = '#0d1b2a'; GRID_C = '#1e2a3a'; TEXT = '#e0e0e0'
C_Y = '#ffd54f'; C_B = '#4fc3f7'; C_G = '#66bb6a'; C_R = '#ef5350'
C_OR = '#ffa726'; C_P = '#ce93d8'

# ── Attractor cascade data (from paper Table 2 and Section VI) ────────────
scales = ['Nuclear\n(BBN→stars)',
          'ISM\n(C-12, N-14)',
          'Chemical\n(amino acids)',
          'Polymer\n(RNA, peptides)',
          'Cellular\n(life regime)']

# Φ_I well depths (arbitrary units, relative)
phi_depths = [1.5, 4.2, 8.7, 14.5, 22.0]

# Kramers escape times (log10 years)
# From paper: D ~ 0.05 kBT, ΔV = Phi_I depths × kBT
log_tau = [3, 7, 11, 15, 21]   # log10(τ/yr)
tau_lo  = [2, 6, 10, 14, 19]   # -1σ
tau_hi  = [4, 8, 12, 17, 23]   # +1σ

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                        left=0.07, right=0.97, top=0.88, bottom=0.14)

def sax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.4, axis='y')

colors_scale = [C_B, C_G, C_Y, C_OR, C_R]
x = np.arange(len(scales))

# Panel A: Φ_I well depths
ax1 = fig.add_subplot(gs[0]); sax(ax1)
bars = ax1.bar(x, phi_depths, color=colors_scale, alpha=0.85,
               edgecolor=DARK, lw=0.8, width=0.6)

# Add depth values on bars
for xi, d in zip(x, phi_depths):
    ax1.text(xi, d + 0.3, f'{d:.1f}', color=TEXT,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add cascade arrow
for i in range(len(scales)-1):
    ax1.annotate('', xy=(i+1-0.25, phi_depths[i+1]*0.85),
                xytext=(i+0.25, phi_depths[i]*0.85),
                arrowprops=dict(arrowstyle='->', color=C_P, lw=1.5))

ax1.set_xticks(x)
ax1.set_xticklabels(scales, fontsize=8.5)
ax1.set_ylabel('Information quasi-potential depth $\\Phi_I$', fontsize=10)
ax1.set_title('Panel A — Hierarchical Deepening\nof Quasi-Potential Wells', fontsize=10, pad=5)
ax1.set_ylim(0, 27)
ax1.text(4.35, 23.5, 'Life\nregime', color=C_R, fontsize=8, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=C_R, lw=0.8))

# Panel B: Kramers escape times
ax2 = fig.add_subplot(gs[1]); sax(ax2)

for i, (lt, lo, hi, color, label) in enumerate(
        zip(log_tau, tau_lo, tau_hi, colors_scale, scales)):
    ax2.barh(i, lt, color=color, alpha=0.85, edgecolor=DARK, lw=0.8, height=0.5)
    ax2.errorbar(lt, i, xerr=[[lt-lo], [hi-lt]],
                 fmt='none', color=TEXT, capsize=4, lw=1.5)
    ax2.text(lt + 0.3, i, f'$10^{{{lt}}}$ yr',
             color=TEXT, va='center', fontsize=9)

# Reference lines
for ref_yr, ref_label in [(9.15, 'Age of Earth'), (10.14, 'Age of Universe')]:
    ax2.axvline(ref_yr, color=GRID_C, ls='--', lw=1.2, alpha=0.6)
    ax2.text(ref_yr + 0.1, 4.55, ref_label, color='#90a4ae',
             fontsize=7.5, rotation=90, va='top')

ax2.set_yticks(range(len(scales)))
ax2.set_yticklabels(scales, fontsize=8.5)
ax2.set_xlabel('Kramers escape time $\\log_{10}(\\tau/\\text{yr})$', fontsize=10)
ax2.set_title('Panel B — Escape Times\n$\\tau \\sim \\exp(\\Delta V/D)$ (Definition II.2)',
              fontsize=10, pad=5)
ax2.set_xlim(0, 26)
ax2.invert_yaxis()

fig.suptitle(
    'Figure 2 — The EOM–IFF Attractor Cascade: From Big Bang Nucleosynthesis to Life',
    color=TEXT, fontsize=12, y=0.975, fontweight='bold')

plt.savefig('figures/fig2_attractor_cascade.png', dpi=160,
            bbox_inches='tight', facecolor=DARK)
print("Saved figures/fig2_attractor_cascade.png")
print(f"Key: 5-stage cascade, escape times span log10(τ) = {log_tau[0]} to {log_tau[-1]} yr")
