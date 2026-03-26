"""
Script 02 — Blank (2001) Two-Stage Statistical Analysis
========================================================
Reproduces the complete two-stage analysis of Blank et al. (2001)
shock-synthesis data reported in Section X.A and SI Section S2.

Stage 1 — Independent prediction:
  P*_pred = C * (DeltaPhi_I / sqrt(2D))^1.25
  Inputs: DeltaPhi_I (Markov chain), D (Miyakawa 2002), C (Hugoniot EOS)
  No data from Blank (2001) used.
  Result: P*_pred = 24.8 +/- 4.9 GPa, 95% CI [16.9, 36.0] GPa

Stage 2 — Statistical validation:
  Gaussian fit Y(P) = A * exp(-(P-P*)^2 / (2w^2))
  chi2_red = 4.99 => sigma rescaled by sqrt(4.99) = 2.23
  Result: P*_fit = 28.4 +/- 1.4 GPa, R^2 = 0.885
  Delta_AIC = 34.9 (Gaussian vs linear, very strong evidence)
  Bootstrap 95% CI: [21.3, 34.8] GPa
  LOO range: [26.4, 29.3] GPa

Usage:
  python scripts/02_blank_two_stage.py

Dependencies: numpy>=1.20, scipy>=1.7
"""

import numpy as np
from scipy.optimize import curve_fit
np.random.seed(42)


# ── Blank (2001) data ─────────────────────────────────────────────────────
PRESSURE = np.array([5, 10, 15, 21, 25, 32, 42, 55], dtype=float)  # GPa
YIELD    = np.array([0.0010, 0.0030, 0.0180, 0.0450,
                     0.0410, 0.0380, 0.0150, 0.0050])               # nmol
SIGMA_Y  = np.array([0.0003, 0.0006, 0.0027, 0.0054,
                     0.0049, 0.0046, 0.0030, 0.0015])               # nmol
N_DATA   = len(PRESSURE)


def gaussian(P, A, Pstar, w):
    return A * np.exp(-(P - Pstar)**2 / (2 * w**2))


def linear_model(P, a, b):
    return a + b * P


# ── Stage 1: Independent prediction ──────────────────────────────────────
def stage1_prediction(n_mc=200_000):
    """
    Predict P* from first principles without using Blank (2001) data.

    Inputs (all independent of Blank 2001):
      DeltaPhi_I = 0.450 +/- 0.035  (five-state Markov chain, Script 01)
      E_a        = 50 +/- 10 kJ/mol  (Strecker synthesis, Miyakawa 2002)
      C          = 15.5 +/- 1.5 GPa  (water Hugoniot EOS, Marsh 1980)
    """
    kBT       = 8.314e-3 * 298.0        # kJ/mol at 25 degC
    D_central = kBT / 50.0              # = kBT/Ea
    D_sigma   = kBT * 10.0 / 50.0**2   # propagated uncertainty

    dPhi_c,  dPhi_s = 0.450, 0.035
    C_c,     C_s    = 15.5,  1.5

    def pred(dPhi, D, C):
        return C * (dPhi / np.sqrt(2 * D))**1.25

    # Central value
    P_central = pred(dPhi_c, D_central, C_c)

    # Monte Carlo uncertainty propagation
    dPhi_mc = np.random.normal(dPhi_c, dPhi_s, n_mc)
    D_mc    = np.random.normal(D_central, D_sigma, n_mc)
    C_mc    = np.random.normal(C_c, C_s, n_mc)
    mask    = (dPhi_mc > 0.2) & (D_mc > 0.01) & (C_mc > 8.0)
    P_mc    = pred(dPhi_mc[mask], D_mc[mask], C_mc[mask])

    return {
        'central':    P_central,
        'mean':       np.mean(P_mc),
        'std':        np.std(P_mc),
        'ci68':       np.percentile(P_mc, [16, 84]),
        'ci95':       np.percentile(P_mc, [2.5, 97.5]),
        'sigma_frac': np.std(P_mc) / np.mean(P_mc),
        'n_valid':    mask.sum(),
    }


# ── Stage 2: Statistical validation ──────────────────────────────────────
def stage2_fit():
    """
    Gaussian fit with honest chi2_red reporting and sigma rescaling.
    """
    # Raw fit (absolute sigma)
    popt_raw, pcov_raw = curve_fit(
        gaussian, PRESSURE, YIELD,
        p0=[0.048, 27, 9.1],
        sigma=SIGMA_Y, absolute_sigma=True
    )
    Y_pred_raw = gaussian(PRESSURE, *popt_raw)
    chi2_raw   = np.sum(((YIELD - Y_pred_raw) / SIGMA_Y)**2)
    dof        = N_DATA - 3
    chi2_red   = chi2_raw / dof
    scale      = np.sqrt(chi2_red)

    # Rescaled fit
    sigma_eff = SIGMA_Y * scale
    popt, pcov = curve_fit(
        gaussian, PRESSURE, YIELD,
        p0=[0.048, 28, 8],
        sigma=sigma_eff, absolute_sigma=True
    )
    perr = np.sqrt(np.diag(pcov))
    A_fit, Pstar_fit, w_fit = popt
    A_err, Pstar_err, w_err = perr

    Y_pred = gaussian(PRESSURE, *popt)
    SS_res = np.sum((YIELD - Y_pred)**2)
    SS_tot = np.sum((YIELD - YIELD.mean())**2)
    R2     = 1 - SS_res / SS_tot

    # Standardised residuals (raw sigma)
    resid_std = (YIELD - Y_pred) / SIGMA_Y

    return {
        'A':          A_fit,    'A_err':     A_err,
        'Pstar':      Pstar_fit,'Pstar_err': Pstar_err,
        'w':          w_fit,    'w_err':     w_err,
        'chi2_red':   chi2_red, 'scale':     scale,
        'R2':         R2,
        'sigma_eff':  sigma_eff,
        'resid_std':  resid_std,
        'popt':       popt,
    }


def aic_comparison(sigma_eff, popt_gauss):
    """Compute AIC for Gaussian vs linear model (rescaled sigma)."""
    def aic(model, k, p0):
        po, _ = curve_fit(model, PRESSURE, YIELD,
                           p0=p0, sigma=sigma_eff, absolute_sigma=True)
        Yf    = model(PRESSURE, *po)
        chi2  = np.sum(((YIELD - Yf) / sigma_eff)**2)
        logL  = (-0.5 * chi2
                 - 0.5 * N_DATA * np.log(2 * np.pi)
                 - np.sum(np.log(sigma_eff)))
        return 2 * k - 2 * logL

    aic_g = aic(gaussian,     3, popt_gauss)
    aic_l = aic(linear_model, 2, [0.02, 0.0])
    return aic_g, aic_l, aic_l - aic_g


def bootstrap_ci(sigma_eff, popt, n_boot=15_000):
    """Bootstrap 95% CI for P*."""
    boot_pstar = []
    for _ in range(n_boot):
        idx  = np.random.choice(N_DATA, N_DATA, replace=True)
        Yb   = YIELD[idx] + np.random.normal(0, sigma_eff[idx])
        Yb   = np.maximum(Yb, 1e-7)
        try:
            pb, _ = curve_fit(gaussian, PRESSURE[idx], Yb,
                               p0=popt, sigma=sigma_eff[idx],
                               absolute_sigma=True)
            if 10 < pb[1] < 55 and pb[2] > 0:
                boot_pstar.append(pb[1])
        except Exception:
            pass
    boot_arr = np.array(boot_pstar)
    return np.percentile(boot_arr, [2.5, 97.5]), len(boot_arr)


def loo_crossval(sigma_eff, popt):
    """Leave-one-out cross-validation for P*."""
    loo_pstar = []
    for i in range(N_DATA):
        mask = np.arange(N_DATA) != i
        try:
            pb, _ = curve_fit(gaussian,
                               PRESSURE[mask], YIELD[mask],
                               p0=popt, sigma=sigma_eff[mask],
                               absolute_sigma=True)
            loo_pstar.append(pb[1])
        except Exception:
            loo_pstar.append(np.nan)
    arr = np.array(loo_pstar)
    return arr


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 66)
    print("  Blank (2001) Two-Stage Analysis — Verified Parameters")
    print("=" * 66)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    s1 = stage1_prediction()
    print("\n--- Stage 1: Independent Prediction ---")
    print(f"  Central value : {s1['central']:.1f} GPa")
    print(f"  MC mean       : {s1['mean']:.1f} +/- {s1['std']:.1f} GPa  (1-sigma)")
    print(f"  68% CI        : [{s1['ci68'][0]:.1f}, {s1['ci68'][1]:.1f}] GPa")
    print(f"  95% CI        : [{s1['ci95'][0]:.1f}, {s1['ci95'][1]:.1f}] GPa")
    print(f"  sigma/mean    : {s1['sigma_frac']:.3f}  (<0.30 => determinate prediction)")
    print(f"  n_valid draws : {s1['n_valid']:,} / 200,000")

    # ── Stage 2 ───────────────────────────────────────────────────────────
    s2 = stage2_fit()
    print("\n--- Stage 2: Statistical Validation ---")
    print(f"  chi2_red (raw)         : {s2['chi2_red']:.4f}")
    print(f"  sigma rescaling factor : {s2['scale']:.4f}  (= sqrt(chi2_red))")
    print(f"  A     = {s2['A']:.4f} +/- {s2['A_err']:.4f} nmol")
    print(f"  P*    = {s2['Pstar']:.2f}  +/- {s2['Pstar_err']:.2f}  GPa")
    print(f"  w     = {s2['w']:.2f}  +/- {s2['w_err']:.2f}  GPa")
    print(f"  R^2   = {s2['R2']:.4f}")

    print("\n  Standardised residuals (raw sigma):")
    for i in range(N_DATA):
        flag = "  <-- |r|>2" if abs(s2['resid_std'][i]) > 2 else ""
        print(f"    P={PRESSURE[i]:3.0f} GPa:  r = {s2['resid_std'][i]:+.3f}{flag}")

    # ── AIC comparison ────────────────────────────────────────────────────
    aic_g, aic_l, delta_aic = aic_comparison(s2['sigma_eff'], s2['popt'])
    print(f"\n  AIC comparison (rescaled sigma):")
    print(f"    AIC_Gaussian = {aic_g:.2f}  (k=3)")
    print(f"    AIC_linear   = {aic_l:.2f}  (k=2)")
    print(f"    delta_AIC    = {delta_aic:.1f}  (>10 => very strong evidence for Gaussian)")

    # ── Bootstrap ─────────────────────────────────────────────────────────
    print("\n  Bootstrap 95% CI for P* (n=15,000 draws):")
    ci_boot, n_conv = bootstrap_ci(s2['sigma_eff'], s2['popt'])
    print(f"    [{ci_boot[0]:.1f}, {ci_boot[1]:.1f}] GPa  ({n_conv} converged)")

    # ── LOO ───────────────────────────────────────────────────────────────
    loo = loo_crossval(s2['sigma_eff'], s2['popt'])
    print(f"\n  Leave-one-out P* values:")
    for i in range(N_DATA):
        print(f"    drop P={PRESSURE[i]:3.0f} GPa:  P*_LOO = {loo[i]:.2f} GPa")
    print(f"    Range: [{np.nanmin(loo):.2f}, {np.nanmax(loo):.2f}] GPa  "
          f"(std = {np.nanstd(loo):.3f})")

    # ── Consistency check ─────────────────────────────────────────────────
    print("\n  Consistency: Stage 1 95% CI contains Stage 2 P*_fit?")
    lo, hi = s1['ci95']
    contains = lo < s2['Pstar'] < hi
    print(f"    {lo:.1f} < {s2['Pstar']:.1f} < {hi:.1f}  =>  "
          f"{'YES, consistent' if contains else 'NO, inconsistent'}")

    # ── Paper value verification ───────────────────────────────────────────
    print("\n  Verification against paper values (Section X.A / SI S2):")
    checks = [
        ("P*_pred MC mean", s1['mean'],       24.8, 0.3),
        ("95% CI lower",    s1['ci95'][0],    16.9, 0.5),
        ("95% CI upper",    s1['ci95'][1],    36.0, 0.5),
        ("P*_fit",          s2['Pstar'],      28.4, 0.1),
        ("P*_fit err",      s2['Pstar_err'],   1.4, 0.1),
        ("w_fit",           s2['w'],           8.3, 0.1),
        ("R^2",             s2['R2'],          0.885, 0.002),
        ("chi2_red",        s2['chi2_red'],    4.99, 0.05),
        ("delta_AIC",       delta_aic,        34.9, 0.5),
    ]
    all_ok = True
    for name, val, exp, tol in checks:
        ok = abs(val - exp) <= tol
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: {val:.3f}  (paper: {exp})")

    print()
    if all_ok:
        print("  RESULT: All values match paper exactly.")
    else:
        print("  WARNING: Mismatch — check scipy/numpy version.")


if __name__ == "__main__":
    main()
