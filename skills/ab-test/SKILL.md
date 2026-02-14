---
name: ab-test
description: A comprehensive guide for designing, analyzing, and decision-making in A/B testing (Online Controlled Experiments). Use this skill for sample size calculation, power analysis, statistical significance testing, Bayesian analysis, or when users need to interpret experiment results.
last-updated: 2026-01-26
---

# A/B Testing Analysis & Design

This skill provides a rigorous framework for Online Controlled Experiments (OCE), combining methodology based on Kohavi's principles with practical calculation tools.

## When to Use This Skill

- Calculate sample sizes for A/B tests
- Test statistical significance between variants
- Perform power analysis
- Run Bayesian analysis (probability to beat baseline)
- Validate experiment integrity (SRM, peeking)
- Make Ship/No-Ship decisions

## Quick Start: Calculator

```python
from ab_test_calc import ABTestCalculator

calc = ABTestCalculator()

# Test significance
result = calc.test_significance(
    control_visitors=10000,
    control_conversions=500,
    variant_visitors=10000,
    variant_conversions=550
)

print(f"Significant: {result['significant']}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Lift: {result['lift']:.2%}")
```

### CLI Usage

```bash
# Test significance
python ab_test_calc.py --test 10000 500 10000 550

# Calculate sample size
python ab_test_calc.py --sample-size --baseline 0.05 --mde 0.10 --power 0.8

# Power analysis
python ab_test_calc.py --power-analysis --baseline 0.05 --mde 0.10 --samples 5000

# Bayesian analysis
python ab_test_calc.py --bayesian 10000 500 10000 550
```

## 1. Experiment Design & Sizing

Before running an experiment, determine if the design has sufficient statistical power.

### Minimum Detectable Effect (MDE)

The MDE is a **business constraint**, not a statistical output. It answers: *What is the smallest lift that justifies the cost of this feature?*

- Sample size is inversely proportional to MDE². Halving the MDE requires 4× the traffic.
- Do not set MDE lower than "Practical Significance" (e.g., a 0.1% lift might be statistically significant but ROI negative).

### Sample Size Calculation

```python
result = calc.calculate_sample_size(
    baseline_rate=0.05,              # Current 5% conversion
    minimum_detectable_effect=0.10,  # 10% relative improvement
    power=0.8,                       # 80% power
    alpha=0.05                       # 5% significance level
)

print(f"Need {result['sample_size_per_variant']:,} visitors per variant")
print(f"Total: {result['total_sample_size']:,} visitors")
```

### Duration Estimation

```python
duration = calc.estimate_duration(
    daily_visitors=5000,
    baseline_rate=0.03,
    minimum_detectable_effect=0.15
)
print(f"Test will take ~{duration['days']} days")
```

## 2. Pre-Analysis Validation (Sanity Checks)

**CRITICAL:** Before interpreting any metric movements, validate experiment integrity.

### Sample Ratio Mismatch (SRM)

Check if actual user ratio matches design (e.g., 50/50).

- **Diagnosis:** Chi-Square Goodness-of-Fit test
- **Threshold:** If p < 0.01, assume SRM exists
- **Action:** Discard the experiment. Do not try to "fix" the data.

### Twyman's Law

*"Any figure that looks interesting or different is usually wrong."*

If a metric moves by a massive amount (e.g., +50% conversion in a mature product), assume it is a data error until proven otherwise.

See [references/PITFALLS_AND_CHECKS.md](references/PITFALLS_AND_CHECKS.md) for complete checklist.

## 3. Statistical Inference

### Frequentist Analysis (Default)

Best for rigorous binary decisions (Ship/No Ship) with controlled error rates.

```python
result = calc.test_significance(
    control_visitors=10000,
    control_conversions=500,
    variant_visitors=10000,
    variant_conversions=550,
    test="chi_square"  # or "z_test"
)

# Returns:
{
    "significant": True,
    "p_value": 0.0234,
    "control_rate": 0.05,
    "variant_rate": 0.055,
    "lift": 0.10,
    "confidence_interval": {"lower": 0.02, "upper": 0.18},
    "recommendation": "Variant shows significant improvement"
}
```

### Bayesian Analysis

Best when you need "What is the probability B is better than A?"

```python
result = calc.bayesian_analysis(
    control_visitors=10000,
    control_conversions=500,
    variant_visitors=10000,
    variant_conversions=550
)

# Returns:
{
    "prob_variant_better": 0.9523,
    "prob_control_better": 0.0477,
    "expected_lift": 0.098,
    "credible_interval_95": [0.02, 0.18]
}
```

| Framework | Best Used When | Key Metric |
|-----------|----------------|------------|
| **Frequentist** | Rigorous binary decision, high-risk changes | P-value, Confidence Interval |
| **Bayesian** | "Probability B beats A?", optional stopping | Probability to be Best, Expected Loss |

See [references/STATISTICAL_FRAMEWORKS.md](references/STATISTICAL_FRAMEWORKS.md) for detailed methodology.

## 4. Multiple Variant Testing

Test multiple variants with correction for multiple comparisons:

```python
result = calc.test_multiple_variants(
    control=(10000, 500),
    variants=[
        (10000, 550),  # Variant A
        (10000, 520),  # Variant B
        (10000, 480)   # Variant C
    ],
    correction="bonferroni"  # or "holm", "none"
)

print(f"Winner: {result['winner']}")
```

## 5. Decision Framework

Decision-making should not rely solely on statistical significance.

### Guardrails

Check "Non-Negotiable" metrics (e.g., Latency, Error Rates). If these degrade beyond threshold, kill the experiment regardless of wins elsewhere.

### Common Decision Errors

- **Peeking:** Checking p-values daily inflates False Positives from 5% to ~19%. Use Sequential Testing or Bayesian methods if early stopping is required.
- **Post-Hoc Power:** Never calculate power *after* the experiment based on observed lift.

## API Reference

### ABTestCalculator Class

```python
class ABTestCalculator:
    def __init__(self, alpha: float = 0.05)

    # Significance testing
    def test_significance(self, control_visitors, control_conversions,
                         variant_visitors, variant_conversions,
                         test="chi_square") -> dict

    # Sample size calculation
    def calculate_sample_size(self, baseline_rate, minimum_detectable_effect,
                             power=0.8, alpha=0.05) -> dict

    # Power analysis
    def calculate_power(self, baseline_rate, minimum_detectable_effect,
                       sample_size, alpha=0.05) -> dict

    # Confidence interval
    def confidence_interval(self, visitors, conversions,
                           confidence=0.95) -> dict

    # Bayesian analysis
    def bayesian_analysis(self, control_visitors, control_conversions,
                         variant_visitors, variant_conversions,
                         simulations=100000) -> dict

    # Multiple variants
    def test_multiple_variants(self, control, variants,
                              correction="bonferroni") -> dict

    # Duration estimation
    def estimate_duration(self, daily_visitors, baseline_rate,
                         minimum_detectable_effect, power=0.8) -> dict
```

## Example Workflows

### Pre-Test Planning

```python
calc = ABTestCalculator()

# 1. Estimate required sample size
sample = calc.calculate_sample_size(
    baseline_rate=0.03,
    minimum_detectable_effect=0.15,
    power=0.8
)
print(f"Need {sample['sample_size_per_variant']:,} visitors per variant")

# 2. Estimate test duration
duration = calc.estimate_duration(
    daily_visitors=5000,
    baseline_rate=0.03,
    minimum_detectable_effect=0.15
)
print(f"Test will take ~{duration['days']} days")
```

### Post-Test Analysis

```python
calc = ABTestCalculator()

# 1. Test significance
result = calc.test_significance(15000, 450, 15000, 525)

# 2. Get Bayesian probability
bayes = calc.bayesian_analysis(15000, 450, 15000, 525)

print(f"P-value: {result['p_value']:.4f}")
print(f"Lift: {result['lift']:.2%}")
print(f"Probability variant wins: {bayes['prob_variant_better']:.1%}")
```

## Reference Documentation

- [references/STATISTICAL_FRAMEWORKS.md](references/STATISTICAL_FRAMEWORKS.md) - Frequentist and Bayesian methodology
- [references/PITFALLS_AND_CHECKS.md](references/PITFALLS_AND_CHECKS.md) - Validity threats and sanity checks

## Dependencies

```
scipy>=1.10.0
numpy>=1.24.0
```

Install: `pip install scipy numpy`
