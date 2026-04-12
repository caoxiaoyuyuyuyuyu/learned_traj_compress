"""
Post-hoc power analysis for DPO vs SFT (exp_006, N=8).

Bootstrap CI for delta: [-0.009, 0.029] → width = 0.038
SE = 0.038 / (2 * 1.96) ≈ 0.00969
sigma_diff = SE * sqrt(125) ≈ 0.1084
"""

import numpy as np
from scipy.stats import norm

# Parameters from exp_006
n = 125  # paired prompts
alpha = 0.05
z_alpha = norm.ppf(1 - alpha / 2)  # 1.96 for two-sided

# Estimate sigma_diff from bootstrap CI
ci_low, ci_high = -0.009, 0.029
ci_width = ci_high - ci_low  # 0.038
se = ci_width / (2 * z_alpha)
sigma_diff = se * np.sqrt(n)

print(f"Estimated parameters:")
print(f"  n = {n}")
print(f"  CI width = {ci_width:.3f}")
print(f"  SE = {se:.5f}")
print(f"  sigma_diff = {sigma_diff:.4f} ({sigma_diff*100:.1f}pp)")
print(f"  alpha = {alpha} (two-sided)")
print()

# Power calculation: power = P(|Z| > z_alpha) under H1
# For two-sided test: power = 1 - Phi(z_alpha - delta/SE) + Phi(-z_alpha - delta/SE)
def power_paired(effect, n, sigma_diff, alpha=0.05):
    se = sigma_diff / np.sqrt(n)
    z_crit = norm.ppf(1 - alpha / 2)
    ncp = effect / se  # non-centrality parameter
    power = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
    return power

# Power table
effects_pp = [5, 10, 15, 20]
print("=" * 55)
print(f"{'True effect':>12} {'n':>5} {'sigma_diff':>11} {'Power':>8}")
print("-" * 55)
for eff_pp in effects_pp:
    eff = eff_pp / 100
    pwr = power_paired(eff, n, sigma_diff)
    print(f"{eff_pp:>10}pp {n:>5} {sigma_diff*100:>9.1f}pp {pwr:>8.3f}")
print("=" * 55)
print()

# Detection floor: smallest effect detectable with 80% power at n=125
def required_effect_for_power(target_power, n, sigma_diff, alpha=0.05):
    z_crit = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(target_power)
    se = sigma_diff / np.sqrt(n)
    effect = (z_crit + z_beta) * se
    return effect

min_effect = required_effect_for_power(0.80, n, sigma_diff)
print(f"Detection floor (80% power, n={n}): {min_effect*100:.1f}pp")
print()

# Required n for 80% power at various effect sizes
def required_n(effect, sigma_diff, target_power=0.80, alpha=0.05):
    z_crit = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(target_power)
    n = ((z_crit + z_beta) * sigma_diff / effect) ** 2
    return int(np.ceil(n))

print("Required n for 80% power:")
print("-" * 35)
for eff_pp in effects_pp:
    eff = eff_pp / 100
    req_n = required_n(eff, sigma_diff)
    print(f"  {eff_pp:>2}pp effect → n = {req_n}")
print()

# Also compute for 5pp at 90% power
req_n_90 = required_n(0.05, sigma_diff, target_power=0.90)
print(f"Required n for 90% power at 5pp: {req_n_90}")
