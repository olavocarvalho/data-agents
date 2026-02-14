# Statistical Frameworks Reference

This document details the mathematical engines for Frequentist and Bayesian analysis in A/B testing.

## 1. Frequentist Inference (NHST)

The industry standard for rigorous, binary decision-making. It assumes parameters are fixed and data is random.

### Sample Size Formula (Two-Tailed)

To calculate the minimum sample size  per variant:

**Variables:**

* : Sample size per variant.
* : Pooled variance. For proportions (Conversion Rate ), .
* : Critical value for significance (1.96 for ).
* : Critical value for power (0.84 for Power=80%).
* : Minimum Detectable Effect (absolute difference, e.g., 0.01 for 1%).



### Analysis Logic

* **Null Hypothesis ():** .
* **P-Value:** The probability of observing a difference as extreme as the one seen, assuming  is true.
* **Confidence Interval:** . If the interval includes 0, result is not significant.

## 2. Bayesian Inference

A probabilistic approach that updates "beliefs" based on data. It assumes data is fixed and parameters are probabilistic.

### Core Metrics

Unlike Frequentist p-values, Bayesian methods answer direct business questions :

* **Probability to Be Best ():** The probability that the Treatment is truly better than Control.
* **Expected Loss:** The average metric loss incurred if we choose a variant that is actually worse.

### Conjugate Priors

Computation is simplified using conjugate priors where the posterior is in the same family as the prior.

* **Conversion (Binary Data):** Uses **Beta-Binomial**.
* Prior: 
* Posterior: .


* **Count Data (e.g., Clicks):** Uses **Gamma-Poisson**.

### Advantages vs. Frequentist

* **Optional Stopping:** Valid to check results continuously. You can stop as soon as Expected Loss is below the risk threshold (e.g., risk < $100).


* **Small Samples:** Often more efficient (up to 75% faster) if informative priors are used.


* **Intuitiveness:** Stakeholders understand "95% chance B is better" better than "p-value = 0.04".

