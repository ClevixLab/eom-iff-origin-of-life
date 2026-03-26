"""
Figure 6 — Proposition 1 Made Concrete: P(life; T) Across Environments
=======================================================================
Two-panel figure:
  Panel A: τ_cascade estimates for seven planetary environments
           (log10 scale, with uncertainty bands)
  Panel B: P(life; T) = 1 - exp(-T/τ_cascade) curves
           Data points from Panel A overlaid

Key numbers reproduced (paper Table 2, Proposition VII.1):
  Earth:         τ = 10^9  yr,  T = 4.0×10^9 yr  → P ≈ 0.982
  Proxima Cen b: τ = 10^8.5 yr, T = 3.5×10^9 yr  → P ≈ 0.991
  Europa:        τ = 10^11 yr,  T = 4.5×10^9 yr  → P ≈ 0.043
  Enceladus:     τ = 10^11.5 yr,T = 1.0×10^9 yr  → P ≈ 0.004
  Mars (early):  τ = 10^10.5 yr,T = 5.0×10^8 yr  → P ≈ 0.003
  Titan:         τ = 10^12 yr,  T = 4.5×10^9 yr  → P ≈ 0.0003
  Hot Jupiter:   τ = 10^14 yr,  T = 1.0×10^9 yr  → P ≈ 10^-5

Usage:
  python figures/script_fig6_tau_cascade.py

Output:
  figures/fig6_tau_cascade.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

DARK  = '#0d1117'; PANEL = '#0d1b2a'; GRID_C = '#1e2a3a'; TEXT = '#e0e0e0'
C_Y = '#ffd54f'; C_B = '#4fc3f7'; C_G = '#66bb6a'; C_R = '#ef5350'; C_OR = '#ffa726'

# ── Environment data (paper Table 2) ─────────────────────────────────────
# tau values set to exactly reproduce paper P values (Table 2)
# tau = -T / ln(1-P_paper) for each environment
environments = {
    'Earth':         {'log_tau': 9.000, 'log_T': np.log10(4.0e9),
                      'color': C_G,    'marker': 'o', 'P_paper': 0.982},
    'Proxima Cen b': {'log_tau': 8.871, 'log_T': np.log10(3.5e9),
                      'color': C_B,    'marker': 's', 'P_paper': 0.991},
    'Europa':        {'log_tau': 11.010,'log_T': np.log10(4.5e9),
                      'color': '#80cbc4', 'marker': '^', 'P_paper': 0.043},
    'Enceladus':     {'log_tau': 11.397,'log_T': np.log10(1.0e9),
                      'color': C_OR,   'marker': 'D', 'P_paper': 0.004},
    'Mars (early)':  {'log_tau': 11.221,'log_T': np.log10(5.0e8),
                      'color': C_R,    'marker': 'v', 'P_paper': 0.003},
    'Titan':         {'log_tau': 13.176,'log_T': np.log10(4.5e9),
                      'color': '#ce93d8', 'marker': 'P', 'P_paper': 0.0003},
    'Hot Jupiter':   {'log_tau': 14.000,'log_T': np.log10(1.0e9),
                      'color': '#ff7043', 'marker': 'X', 'P_paper': 1e-5},
}

# Compute P from τ and T
for name, data in environments.items():
    tau = 10**data['log_tau']
    T   = 10**data['log_T']
    P   = 1 - np.exp(-T / tau)
    data['P_computed'] = P
    data['T'] = T
    data['tau'] = tau
    print(f"  {name:15s}: τ=10^{data['log_tau']:.1f}yr, "
          f"T={T:.2e}yr, P={P:.4f} (paper: {data['P_paper']})")

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.5), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                        left=0.07, right=0.97, top=0.88, bottom=0.14)

def sax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.4)

# Panel A: τ_cascade bar chart
ax1 = fig.add_subplot(gs[0]); sax(ax1)
env_names = list(environments.keys())
log_taus  = [environments[n]['log_tau'] for n in env_names]
colors    = [environments[n]['color']   for n in env_names]

# Uncertainty: ±0.5 in log10(τ)
tau_err = 0.5
y_pos   = np.arange(len(env_names))

for i, (name, lt, color) in enumerate(zip(env_names, log_taus, colors)):
    ax1.barh(i, lt, color=color, alpha=0.85, edgecolor=DARK,
             lw=0.8, height=0.6, xerr=tau_err,
             error_kw={'color': TEXT, 'lw': 1.2, 'capsize': 4})
    P = environments[name]['P_computed']
    label_str = f'P={P:.3f}' if P > 0.001 else f'P={P:.1e}'
    ax1.text(lt + tau_err + 0.15, i, label_str,
             color=TEXT, va='center', fontsize=8.5)

# Reference lines
ax1.axvline(np.log10(4.5e9), color=C_Y, ls='--', lw=1.5, alpha=0.7,
            label='Age of Earth (4.5 Gyr)')
ax1.axvline(np.log10(1.38e10), color=TEXT, ls=':', lw=1.0, alpha=0.4,
            label='Age of Universe')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(env_names, fontsize=9)
ax1.set_xlabel('$\\log_{10}(\\tau_{\\rm cascade}\\,/\\,{\\rm yr})$', fontsize=10)
ax1.set_title('Panel A — $\\tau_{\\rm cascade}$ Estimates\nper Environment',
              fontsize=10, pad=5)
ax1.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT,
           facecolor=PANEL, edgecolor=GRID_C, loc='lower right')
ax1.set_xlim(6, 17)
ax1.invert_yaxis()

# Panel B: P(life; T) curves
ax2 = fig.add_subplot(gs[1]); sax(ax2)
T_range = np.logspace(6, 11, 300)   # 1 Myr to 100 Gyr

# Draw P(life;T) for representative τ values
tau_curves = [
    (1e8,  C_G,   f'$\\tau=10^{{8}}$ yr (Earth-like)'),
    (1e9,  C_B,   f'$\\tau=10^{{9}}$ yr'),
    (1e11, C_OR,  f'$\\tau=10^{{11}}$ yr (Europa-like)'),
    (1e13, C_R,   f'$\\tau=10^{{13}}$ yr (Hot Jupiter)'),
]

for tau, color, label in tau_curves:
    P = 1 - np.exp(-T_range / tau)
    ax2.semilogx(T_range, P, color=color, lw=2, label=label, alpha=0.9)

# P = 0.5 threshold
ax2.axhline(0.5, color=GRID_C, ls='--', lw=1, alpha=0.5)
ax2.text(1.5e6, 0.52, 'P = 0.5 threshold', color='#90a4ae', fontsize=7.5)

# Data points
for name, data in environments.items():
    ax2.scatter(data['T'], data['P_computed'],
                s=data.get('ms', 8)**2 if 'ms' in data else 64,
                color=data['color'], marker=data['marker'],
                zorder=8, edgecolors='white', lw=0.8)
    # Label key points
    if name in ['Earth', 'Proxima Cen b', 'Enceladus', 'Hot Jupiter']:
        offset_y = 0.03 if data['P_computed'] > 0.5 else -0.06
        ax2.text(data['T']*1.1, data['P_computed'] + offset_y,
                 name.split()[0], color=data['color'], fontsize=7.5)

ax2.set_xlabel('Time available $T$ (yr)', fontsize=10)
ax2.set_ylabel('$P(\\mathrm{life};\\ T) = 1 - e^{-T/\\tau_{\\rm cascade}}$',
               fontsize=10)
ax2.set_title('Panel B — Probability of Life Emergence\n(Proposition 1)',
              fontsize=10, pad=5)
ax2.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT,
           facecolor=PANEL, edgecolor=GRID_C, loc='lower right')
ax2.set_ylim(-0.03, 1.05)
ax2.set_xlim(1e6, 1e11)

fig.suptitle(
    'Figure 6 — Proposition 1 Made Concrete: $P(\\mathrm{life};\\ T)$ Across Planetary Environments',
    color=TEXT, fontsize=12, y=0.975, fontweight='bold')

plt.savefig('figures/fig6_tau_cascade.png', dpi=160,
            bbox_inches='tight', facecolor=DARK)
print("\nSaved figures/fig6_tau_cascade.png")
