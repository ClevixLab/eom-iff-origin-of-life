# eom-iff-origin-of-life

**Computation scripts and figures for:**

> Truong Quynh Hoa & Truong Xuan Khanh (2026).
> *From Big Bang to Biochemistry: Entropy-Oriented Mechanics and Information Force Fields
> as a Unifying Framework for the Origin of Carbon-Based Life.*
> arXiv:cs.AI (submitted).

---

## Repository structure

```
eom-iff-origin-of-life/
├── scripts/                     # Numerical computations
│   ├── 01_markov_chain_pstar.py         # Table I: p*(x) and Φ_I(x)
│   ├── 02_blank_two_stage.py            # Section X.A: two-stage Blank analysis
│   ├── 03_proposition22_verification.py # SI S5: Proposition 2.2 bound
│   └── 04_run_all_and_verify.py         # Master runner — all 28 checks
│
├── figures/                     # Figure generation scripts + PNG outputs
│   ├── script_fig1_framework_validation.py   # Fig 1: polymer yield, distributions, Theorem 1
│   ├── script_fig2_attractor_cascade.py      # Fig 2: Φ_I cascade + Kramers times
│   ├── script_fig3_ferris_validation.py      # Fig 3: Ferris synergy (Prediction II)
│   ├── script_fig4_nuclear_attractors.py     # Fig 4: nuclear Φ_I, CNO cycle
│   ├── script_fig5_phase_diagram.py          # Fig 5: 6-regime phase diagram
│   ├── script_fig6_tau_cascade.py            # Fig 6: τ_cascade + P(life;T)
│   ├── script_fig7_blank_fitting.py          # Fig 7: two-stage Blank fit
│   ├── fig1_framework_validation.png         # Paper figure
│   ├── fig2_attractor_cascade.png
│   ├── fig3_experimental_validation.png
│   ├── fig4_nuclear_attractors.png
│   ├── fig5_phase_diagram.png
│   ├── fig6_tau_cascade.png
│   └── fig7_blank_fitting.png
│
└── SI/                          # Supplemental Information scripts
    ├── script_SI_S3_tau_extended.py     # SI Table S3: 15-environment τ_cascade
    └── script_SI_S4_ferris_sensitivity.py  # SI Table S4: γ sensitivity (exact independence)
```

---

## Quick start

```bash
git clone https://github.com/ClevixLab/eom-iff-origin-of-life.git
cd eom-iff-origin-of-life

pip install -r requirements.txt

# Run full verification suite (should print: 28 passed, 0 failed)
python scripts/04_run_all_and_verify.py

# Regenerate all figures
for script in figures/script_fig*.py; do python $script; done

# Reproduce SI tables
python SI/script_SI_S3_tau_extended.py
python SI/script_SI_S4_ferris_sensitivity.py
```

---

## What each script reproduces

### Numerical results (scripts/)

| Script | Paper location | Key result |
|--------|---------------|------------|
| `01_markov_chain_pstar.py` | Table I | p*(Asp)=0.3254, Φ_I(Asp)=1.1228 |
| `02_blank_two_stage.py` | Section X.A | P*=28.4±1.4 GPa, R²=0.885, ΔAIC=34.9 |
| `03_proposition22_verification.py` | SI Section S5 | Prop 2.2 bound satisfied, P(tetramer>monomer)=0.98 |
| `04_run_all_and_verify.py` | Full paper | 28/28 verified values |

### Figures (figures/)

| Script | Figure | Content |
|--------|--------|---------|
| `script_fig1_framework_validation.py` | Fig 1 | Non-monotonic polymer yield (Prediction I); Theorem 1 numerical check |
| `script_fig2_attractor_cascade.py` | Fig 2 | Φ_I well depths + Kramers escape times across 5 scales |
| `script_fig3_ferris_validation.py` | Fig 3 | Superlinearity factor 4.2, synergy 83% (Prediction II) |
| `script_fig4_nuclear_attractors.py` | Fig 4 | Nuclear Φ_I in stellar core vs ISM; p*(N-14)≈0.95 |
| `script_fig5_phase_diagram.py` | Fig 5 | 6-regime (Φ, d) phase diagram with environment symbols |
| `script_fig6_tau_cascade.py` | Fig 6 | τ_cascade for 7 environments; P(life;T) curves |
| `script_fig7_blank_fitting.py` | Fig 7 | Two-stage Blank fit: Stage 1 CI [16.9,36.0] + Stage 2 P*=28.4 GPa |

### SI tables (SI/)

| Script | SI location | Key result |
|--------|-------------|------------|
| `script_SI_S3_tau_extended.py` | Table S3 | 15-environment τ_cascade with P_lo/P_hi bounds |
| `script_SI_S4_ferris_sensitivity.py` | Table S4 | Synergy 83% and superlinearity 4.2 exactly independent of γ |

---

## Key verified numbers

### Table I — Five-state amino-acid network

| State | p*(x) | Φ_I(x) |
|-------|--------|---------|
| Glycine | 0.2076 | 1.5722 |
| Alanine | 0.2129 | 1.5469 |
| Serine | 0.1220 | 2.1034 |
| **Aspartate** | **0.3254** | **1.1228** ← deepest |
| Degraded | 0.1321 | 2.0241 |

**Solver note:** Direct linear solvers fail (condition number ~10¹⁶).
Scripts use `scipy.linalg.eig` with residual ‖Kp*‖∞ < 10⁻²⁰.

### Two-stage Blank (2001) analysis

| Quantity | Value |
|----------|-------|
| Stage 1 (independent prediction) | P* = 24.8 ± 4.9 GPa, 95% CI [16.9, 36.0] GPa |
| Stage 2 fit (rescaled σ) | P* = 28.4 ± 1.4 GPa, R² = 0.885 |
| χ²_red (before rescaling) | 4.99 |
| ΔAIC vs linear | 34.9 |
| Bootstrap 95% CI | [21.3, 34.8] GPa |
| LOO range | [26.4, 29.3] GPa |

### Ferris (1996) synergy

| Quantity | Value |
|----------|-------|
| Additive prediction L_add | 12 monomers |
| Observed L_both | 50 monomers |
| Superlinearity factor | 50/12 ≈ 4.2 (exact) |
| Synergy fraction | 83% (exact, independent of γ) |

---

## Dependencies

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
```

---

## Notes on figure reproduction

Figures 1–6 are **illustrative** — the scripts reproduce the correct
quantitative relationships and key numbers from the paper, using the
same mathematical model. Minor visual differences from the paper figures
(colour palette, exact layout) are expected since those were refined
for publication. The underlying data and calculations are identical.

Figure 7 is **exact** — it fits the Blank (2001) data with the
verified two-stage procedure and produces the same P*, R², and ΔAIC.

---

## Authors

**Truong Quynh Hoa** (co-first, corresponding) — hoa@clevix.vn  
**Truong Xuan Khanh** (co-first) — khanh@clevix.vn  
H&K Research Studio, Clevix LLC, Hanoi, Vietnam

---

## License

MIT — see `LICENSE`.

---

## Citation

```bibtex
@article{truong2026eomiff,
  title   = {From Big Bang to Biochemistry: Entropy-Oriented Mechanics
             and Information Force Fields as a Unifying Framework
             for the Origin of Carbon-Based Life},
  author  = {Truong, Quynh Hoa and Truong, Xuan Khanh},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv cs.AI}
}
```
