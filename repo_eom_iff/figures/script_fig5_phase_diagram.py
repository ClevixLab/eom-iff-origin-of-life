"""
Figure 5 — EOM-IFF Phase Diagram
==================================
Six physically distinct regimes in the (Φ, d) plane where:
  Φ = normalised entropy flux
  d = ΔV/D = Kramers attractor depth (Definition II.2)

Regimes:
  1. Frozen / no dissipation   (low Φ, any d)
  2. Shallow attractors        (low d, moderate Φ)
  3. Optimal window (life)     (intermediate Φ, high d)
  4. Deep but undriven         (high d, low Φ)
  5. Destructive flux          (very high Φ, any d)
  6. Transition / cascade      (moderate Φ, moderate d)

Real environments overlaid as symbols.
Yellow trajectory: attractor cascade ISM → Hadean → modern cell.

Usage:
  python figures/script_fig5_phase_diagram.py

Output:
  figures/fig5_phase_diagram.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

DARK  = '#0d1117'; PANEL = '#0d1b2a'; GRID_C = '#1e2a3a'; TEXT = '#e0e0e0'
C_Y = '#ffd54f'; C_B = '#4fc3f7'; C_G = '#66bb6a'; C_R = '#ef5350'; C_OR = '#ffa726'

# ── Phase diagram grid ────────────────────────────────────────────────────
phi_log  = np.linspace(-3, 3, 400)   # log10(Φ/Φ_ref)
d_vals   = np.linspace(0, 80, 400)   # ΔV/D
PHI, D   = np.meshgrid(phi_log, d_vals)

# Regime classification based on condensation condition (Eq. 34):
# IFF dominates when β‖∇Φ_I‖ >> sqrt(2D)  <=> d >> d_crit(Φ)
# Life regime: d > d_life = 55 AND Φ in optimal window
phi_opt_log  = 0.0      # optimal at Φ = Φ_ref
phi_lo_log   = -1.5
phi_hi_log   = 1.2
d_life       = 55
d_shallow    = 15

# Create regime map
regime = np.zeros_like(PHI)
# 1: Frozen (Φ very low)
regime[PHI < -2.5] = 1
# 5: Destructive (Φ very high)
regime[PHI > 1.8] = 5
# 2: Shallow attractors
regime[(PHI >= -2.5) & (PHI <= 1.8) & (D < d_shallow)] = 2
# 4: Deep but undriven
regime[(PHI < phi_lo_log) & (D >= d_shallow) & (PHI >= -2.5)] = 4
# 3: Life / optimal window
regime[(PHI >= phi_lo_log) & (PHI <= phi_hi_log) & (D >= d_life)] = 3
# 6: Transition cascade
regime[(PHI >= phi_lo_log) & (PHI <= phi_hi_log) &
       (D >= d_shallow) & (D < d_life)] = 6
# Override destructive
regime[PHI > 1.8] = 5

# ── Color map for regimes ─────────────────────────────────────────────────
regime_colors = {
    0: PANEL,
    1: '#1a237e',   # frozen — deep blue
    2: '#1b5e20',   # shallow — dark green
    3: '#f9a825',   # life/optimal — gold
    4: '#4a148c',   # deep undriven — purple
    5: '#b71c1c',   # destructive — dark red
    6: '#0277bd',   # transition — blue
}
regime_labels = {
    1: 'Frozen\n(no dissipation)',
    2: 'Shallow attractors\n(no condensation)',
    3: 'Life / optimal\nwindow',
    4: 'Deep but\nundriven',
    5: 'Destructive\nflux',
    6: 'Transition /\ncascade zone',
}

# Build RGB image
rgb = np.zeros((*regime.shape, 3))
for r, hex_color in regime_colors.items():
    mask = regime == r
    if mask.any():
        c = matplotlib.colors.to_rgb(hex_color)
        rgb[mask] = c

# ── Environment data points ────────────────────────────────────────────────
envs = {
    'ISM':          (phi_lo_log - 0.3, 20,  C_B,  'o', 10),
    'Hadean ocean': (phi_opt_log - 0.2, 45, C_G,  's', 10),
    'Modern cell':  (phi_opt_log + 0.1, 62, C_Y,  '*', 14),
    'Enceladus':    (phi_lo_log - 0.8, 18,  C_OR, 'D',  9),
    'Hot Jupiter':  (phi_hi_log + 0.4, 5,   C_R,  'X', 10),
    'Europa':       (phi_lo_log - 0.4, 30,  '#80cbc4', '^', 9),
    'Titan':        (phi_lo_log - 1.2, 12,  '#ce93d8', 'v', 9),
}

# Cascade trajectory: ISM → hydrothermal → prebiotic soup → modern cell
traj_phi = np.array([-1.8, -0.5, -0.1, 0.1])
traj_d   = np.array([20,    35,   50,   62])

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8), facecolor=DARK)
ax.set_facecolor(PANEL)

# Phase map
extent = [phi_log[0], phi_log[-1], d_vals[0], d_vals[-1]]
ax.imshow(rgb, origin='lower', aspect='auto', extent=extent, alpha=0.75)

# Boundary lines
ax.axvline(phi_lo_log, color=C_B,  ls='--', lw=1.5, alpha=0.6)
ax.axvline(phi_hi_log, color=C_R,  ls='--', lw=1.5, alpha=0.6)
ax.axvline(0,          color=C_Y,  ls=':',  lw=1.2, alpha=0.5,
           label=r'$\Phi^*$ (optimal)')
ax.axhline(d_life,     color=C_Y,  ls='-',  lw=2.0, alpha=0.8,
           label=f'$d_{{\\rm life}} = {d_life}$ (Definition VI.1)')
ax.axhline(d_shallow,  color=TEXT, ls=':',  lw=1.0, alpha=0.4)

# Region labels
label_positions = {
    1: (-2.7, 40),
    2: (0.0,  6),
    3: (-0.6, 70),
    4: (-2.0, 65),
    5: (2.1,  40),
    6: (-0.6, 35),
}
for r, (lx, ly) in label_positions.items():
    if r in regime_labels:
        ax.text(lx, ly, regime_labels[r], color=TEXT, fontsize=8,
                ha='center', va='center', alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', fc=regime_colors[r],
                          alpha=0.6, ec='none'))

# Cascade trajectory
ax.plot(traj_phi, traj_d, '-', color=C_Y, lw=2.5, alpha=0.9,
        zorder=8, label='Attractor cascade trajectory')
ax.annotate('', xy=(traj_phi[-1], traj_d[-1]),
            xytext=(traj_phi[-2], traj_d[-2]),
            arrowprops=dict(arrowstyle='->', color=C_Y, lw=2.5), zorder=9)

# Environment points
for name, (px, dy, color, marker, ms) in envs.items():
    ax.scatter(px, dy, s=ms**2, color=color, marker=marker,
               zorder=10, edgecolors='white', lw=0.8)
    offset_x = 0.08
    offset_y = 2
    ax.text(px + offset_x, dy + offset_y, name,
            color=TEXT, fontsize=7.5, va='bottom', zorder=11)

ax.set_xlabel('Normalised entropy flux $\\log_{10}(\\Phi/\\Phi_{\\rm ref})$',
              fontsize=11, color=TEXT)
ax.set_ylabel('Attractor depth $d = \\Delta V/D$', fontsize=11, color=TEXT)
ax.set_title('Figure 5 — EOM–IFF Phase Diagram: Six Structural Regimes',
             fontsize=13, color=TEXT, pad=8, fontweight='bold')
ax.tick_params(colors=TEXT)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)

ax.legend(fontsize=8.5, framealpha=0.3, labelcolor=TEXT,
          facecolor=PANEL, edgecolor=GRID_C, loc='upper left')
ax.set_xlim(phi_log[0], phi_log[-1])
ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig('figures/fig5_phase_diagram.png', dpi=160,
            bbox_inches='tight', facecolor=DARK)
print("Saved figures/fig5_phase_diagram.png")
print(f"Life regime: d > {d_life}, Φ ∈ [{phi_lo_log}, {phi_hi_log}] (log scale)")
