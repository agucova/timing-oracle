# timing-oracle Specification (v2)

## 1. Overview

### Problem Statement

Timing side-channel attacks exploit the fact that cryptographic implementations often take different amounts of time depending on secret data. Detecting these leaks is critical for security, but existing tools have significant limitations:

- **DudeCT** and similar t-test approaches compare means, missing distributional differences (e.g., cache effects that only affect upper quantiles)
- **P-values are misinterpreted**: A p-value of 0.01 doesn't mean "1% chance of leak"—it means "1% chance of this data given no leak." These are very different.
- **Practical vs. statistical significance**: With enough samples, even negligible effects become statistically significant—this is power working correctly, but it creates operational problems when you want CI to pass for effects below some threshold of concern
- **CI flakiness**: Tests that pass locally fail in CI due to environmental noise, or vice versa

### Solution

`timing-oracle` addresses these issues with:

1. **Quantile-based statistics**: Compare nine deciles instead of just means, capturing both uniform shifts (branch timing) and tail effects (cache misses)

2. **Two-layer analysis**:
   - Layer 1 (CI Gate): Bounded false positive rate for reliable pass/fail decisions
   - Layer 2 (Bayesian): Posterior probability of a leak, plus effect size estimates

3. **Interpretable output**: "72% probability of a timing leak with ~50ns effect" instead of "$t = 2.34$, $p = 0.019$"

### Design Goals

- **Controlled FPR**: The CI gate controls false positive rate at approximately $\alpha$ (default 1%) under ideal conditions; discrete timers and environmental factors typically make it conservative (FPR below $\alpha$)
- **Interpretable**: Output includes both a probability (0–100%) and the underlying Bayes factor
- **CI-friendly**: Works reliably across environments without manual tuning
- **Fast**: Under 5 seconds for 100k samples on typical hardware
- **Portable**: Handles different timer resolutions via adaptive batching

### Limitations

**Quantile ceiling**: The highest quantile tested is P90. Effects that manifest only in the top 1–5% of samples (e.g., very rare cache misses) may not be detected. For applications where such rare events are critical, consider using a higher sample count or specialized tail-focused analysis.

---

## 2. Statistical Methodology

This section describes the mathematical foundation of timing-oracle: why we use quantile differences instead of means, how the two-layer architecture provides both reliability and interpretability, and how we estimate uncertainty while keeping computational costs reasonable.

### 2.1 Test Statistic: Quantile Differences

We collect timing samples from two classes:
- **Baseline class**: A specific input (e.g., all zeros, matching plaintext)—the "control" condition
- **Sample class**: Randomly sampled inputs—the "varying" condition

Rather than comparing means, we compare the distributions via their deciles. For each class, compute the 10th, 20th, ..., 90th percentiles, yielding two vectors in $\mathbb{R}^9$. The test statistic is their difference:

$$
\Delta = \hat{q}(\text{Baseline}) - \hat{q}(\text{Sample}) \in \mathbb{R}^9
$$

where $\hat{q}_p$ denotes the empirical $p$-th quantile.

**Why quantiles instead of means?**

Timing leaks manifest in different ways:
- **Uniform shift**: A different code path adds constant overhead → all quantiles shift equally
- **Tail effect**: Cache misses occur probabilistically → upper quantiles shift more than lower

A t-test (comparing means) would detect uniform shifts but completely miss tail effects. Consider a cache-timing attack where 90% of operations are fast, but 10% hit a slow path. The mean shifts slightly, but the 90th percentile shifts dramatically. Quantile differences capture both patterns in a single test statistic.

**Why nine deciles specifically?**

This is a bias-variance tradeoff:
- Fewer quantiles (e.g., just the median) would miss distributional structure
- More quantiles (e.g., percentiles) would require estimating a $99 \times 99$ covariance matrix from limited data, introducing severe estimation noise

Nine deciles capture enough structure to distinguish shift from tail patterns while keeping the covariance matrix ($9 \times 9 = 81$ parameters) tractable with typical sample sizes (10k–100k).

**Note on coverage**: Using deciles means effects confined to the top 10% of samples (above P90) may be underdetected. The CI gate's max-statistic will still respond to P90 shifts, but very rare slow-path effects (affecting <10% of samples) require higher sample counts for reliable detection.

**Quantile computation**: We use R-7 linear interpolation (the default in R and NumPy).² For sorted sample $x$ of size $n$:

$$
h = (n - 1) \cdot p
$$

$$
\hat{q}_p = x_{\lfloor h \rfloor} + (h - \lfloor h \rfloor) \cdot (x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor})
$$

Linear interpolation provides smoother estimates than direct order statistics, which matters for small calibration sets.

² Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. R-7 is estimator type 7 in their taxonomy.

### 2.2 Two-Layer Architecture

A key design decision is using two separate statistical tests rather than one:

| Layer | Question | Method | Output |
|-------|----------|--------|--------|
| CI Gate | Should this block my build? | Block permutation threshold | Pass/Fail |
| Bayesian | What's the probability and magnitude? | Bayes factor | $P(\text{leak})$, BF, effect size |

**Why not just use one?**

Each approach has strengths the other lacks:

*Block permutation thresholds* (Layer 1) provide controlled false positive rate. If you set $\alpha = 0.01$, the test will incorrectly reject approximately 1% of the time under the null hypothesis (no leak). This is crucial for CI: you don't want your build randomly failing on safe code. However, permutation gives you only pass/fail—no probability, no effect size.

*Bayesian inference* (Layer 2) gives you a posterior probability ("72% chance of a leak") and an effect estimate ("~50ns uniform shift"). This is far more informative than pass/fail. However, the posterior depends on your prior and model assumptions. It doesn't provide the same FPR control.

By combining both, you get:
- A reliable CI gate that won't flake
- Rich diagnostic information when you need to understand what's happening

They're computed from the same data but answer different questions. In most cases they agree; when they diverge, the gate is authoritative for CI decisions while the Bayesian layer explains magnitude.

### 2.3 Sample Splitting

The Bayesian layer requires estimating a covariance matrix and setting priors. If we estimate these from the same data we then test, we risk overfitting: the prior becomes tuned to the specific noise realization, inflating our confidence.

To avoid this, we split the data temporally (preserving measurement order):

- **Calibration set** (first 30%): Used to estimate covariance $\Sigma_0$ and set the prior scale
- **Inference set** (last 70%): Used for the actual hypothesis test

Why temporal rather than random split? Timing measurements exhibit autocorrelation—nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. A random split would leak information; a temporal split keeps calibration and inference truly independent.

The 30/70 ratio balances two needs: enough calibration data for stable covariance estimation, enough inference data for statistical power. With 100k samples, the inference set still has 70k samples—plenty for precise quantile estimation.

**Stationarity assumption**: This split assumes the noise environment is approximately stationary across the measurement period. A non-stationarity diagnostic (§2.8) checks this assumption and warns when it's violated.

**Minimum sample requirements:**

| Samples per class | Behavior |
|-------------------|----------|
| $\geq 200$ | Normal 30/70 split |
| 100–199 | Warning; results may have reduced reliability |
| 50–99 | Warning; use 50/50 split for more calibration data |
| $< 50$ | Return Unmeasurable |

Below 200 samples, quantile estimation becomes unreliable, especially for extreme deciles (P10, P90). The P90 estimate with n=70 (after 30/70 split of 100 samples) is based on roughly the 7th largest value—highly variable.

### 2.4 Layer 1: CI Gate

The CI gate's job is to answer "is there a statistically significant timing difference?" with controlled false positive rate. We use a block permutation approach that respects the temporal structure of measurements.

**Core idea:**

If there's no timing leak, the Fixed and Random samples come from the same distribution. The class labels are arbitrary—any observed difference $\Delta$ is just sampling noise plus environmental correlation. By permuting the labels while preserving temporal structure, we can estimate the distribution of $\Delta$ under the null—and set thresholds that control false positives.

**Test statistic:**

We use the maximum absolute quantile difference as our test statistic:

$$
M = \max_{p \in \{0.1, 0.2, \ldots, 0.9\}} |\Delta_p|
$$

This is natural: we're asking "is the largest discrepancy unusually large?" rather than testing each quantile separately and correcting for multiple comparisons.

**Algorithm:**

Measurements are stored as an interleaved time series with class labels: $(t_1, c_1), (t_2, c_2), \ldots$ where $c_i \in \{\text{Fixed}, \text{Random}\}$.

1. **Compute observed statistic**: From the actual labels, compute $\Delta$ and $M = \max_p |\Delta_p|$

2. **Block permutation** ($B = 10{,}000$ iterations):
   - Divide the measurement indices into contiguous blocks of length $b$ (same block length as §2.6)
   - For each block, with probability 0.5, flip all labels in that block (Fixed ↔ Random)
   - Recompute quantiles for each pseudo-class from the permuted labels
   - Compute $\Delta^*$ and record $M^* = \max_p |\Delta^*_p|$

3. **Compute threshold** from the max-statistic distribution:

$$
\tau = \text{Quantile}_{1-\alpha}\left( \{M^{*(1)}, \ldots, M^{*(B)}\} \right)
$$

4. **Test**: Reject $H_0$ (declare leak) if $M > \tau$

**Why block permutation instead of i.i.d. bootstrap?**

Timing measurements are autocorrelated—nearby samples share cache state, frequency scaling effects, and environmental noise. Standard i.i.d. bootstrap destroys this correlation structure, leading to underestimated variance and inflated false positive rates.

Block permutation preserves the temporal structure: within each block, the relative timing of samples is unchanged. Only the class labels are (potentially) swapped. This maintains the autocorrelation while still testing the null hypothesis that labels are exchangeable.

**Why max-statistic instead of per-component corrections?**

The 9 quantile differences are positively correlated—they come from the same samples. Traditional multiple-testing corrections (Bonferroni, Šidák) don't account for this correlation and are unnecessarily conservative, reducing power.

The permutation approach directly estimates the null distribution of the statistic we actually use. The correlation structure is automatically captured. This gives tighter thresholds and better power while still controlling FPR at the target level.

**Threshold flooring:**

The raw permutation threshold $\tau_\alpha$ may be too small in two scenarios:

1. **Discrete timers**: If $\tau_\alpha < 1$ tick, random quantization noise triggers false positives
2. **High-resolution timers with large n**: With rdtsc (~0.3ns) and 100k samples, you can detect sub-nanosecond effects that are practically irrelevant

We apply a two-part floor:

$$
\tau = \max(\tau_\alpha, \text{1 tick}, \text{min\_effect\_of\_concern})
$$

The `min_effect_of_concern` (default 10ns) ensures the CI gate only fails for effects you actually care about. This aligns the gate with practical relevance rather than pure statistical detectability.

**FPR control:**

By construction, the probability that $M > \tau$ is approximately $\alpha$ under $H_0$. Two factors make this conservative in practice:

1. **Monte Carlo error** from finite $B$: with $B = 10{,}000$, the SE on the threshold quantile is ~0.1%
2. **Class-size imbalance** from block-flipping: permuted datasets have $O(\sqrt{n})$ deviation from $(n, n)$, which slightly inflates variance estimates

For $n \geq 10{,}000$, actual FPR is typically $0.5\text{–}1\times\alpha$. On discrete timers with threshold flooring (§2.4), the effect is more pronounced—FPR may be substantially below $\alpha$. This conservatism is acceptable for security applications.

### 2.5 Layer 2: Bayesian Inference

The Bayesian layer provides what the CI gate cannot: a probability that there's a leak, and an estimate of how big it is. We use a conjugate Gaussian model that admits closed-form solutions—no MCMC required.

**The model:**

We frame leak detection as model comparison:

- $H_0$ (no leak): $\Delta \sim \mathcal{N}(0, \Sigma_0)$ — the observed quantile differences are pure noise
- $H_1$ (leak): $\Delta \sim \mathcal{N}(X\beta, \Sigma_0)$ — the differences have a systematic component $X\beta$ plus noise

Under $H_1$, $X$ is a design matrix that decomposes effects into interpretable components, and $\beta = (\mu, \tau)^\top$ are the effect magnitudes.

**Covariance estimation:**

$\Sigma_0$ is the covariance of the difference vector $\Delta$ under the null hypothesis. It is estimated directly via paired block bootstrap on the calibration set (see §2.6)—**not** as the sum of marginal covariances $\Sigma_F + \Sigma_R$.

This distinction matters: with interleaved measurements, common-mode noise creates positive covariance between $\hat{q}_F$ and $\hat{q}_R$. The true variance of $\Delta$ is:

$$
\text{Var}(\Delta) = \Sigma_F + \Sigma_R - 2\,\text{Cov}(\hat{q}_F, \hat{q}_R)
$$

Summing marginal covariances would overestimate variance, making the Bayesian layer overly conservative. Bootstrapping $\Delta$ directly captures the correct (smaller) variance.

#### Design Matrix

We want to distinguish two types of timing leaks:

- **Uniform shift** ($\mu$): All quantiles move by the same amount (e.g., a branch that adds constant overhead)
- **Tail effect** ($\tau$): Upper quantiles shift more than lower ones (e.g., cache misses that occur probabilistically)

The design matrix encodes this decomposition:

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix}
$$

where:
- $\mathbf{1} = (1, 1, 1, 1, 1, 1, 1, 1, 1)^\top$ — a uniform shift moves all 9 quantiles equally
- $\mathbf{b}_{\text{tail}}$ — a tail effect basis (see orthogonalization below)

An effect of $\mu = 10, \tau = 0$ means a pure 10ns uniform shift; $\mu = 0, \tau = 20$ means upper quantiles are slower while lower quantiles are faster than the mean.

**Orthogonalization:**

The raw tail basis $\mathbf{b}_{\text{tail}}^{(0)} = (-0.5, -0.375, \ldots, 0.375, 0.5)^\top$ is centered (sums to zero), making it orthogonal to $\mathbf{1}$ under the standard inner product. However, GLS inference uses the $\Sigma_0^{-1}$-weighted inner product, so for $\mu$ and $\tau$ to be uncorrelated, we need:

$$
\mathbf{1}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} = 0
$$

After estimating $\Sigma_0$ from the calibration set, we orthogonalize via one Gram-Schmidt step:

$$
\mathbf{b}_{\text{tail}} = \mathbf{b}_{\text{tail}}^{(0)} - \frac{\mathbf{1}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}}^{(0)}}{\mathbf{1}^\top \Sigma_0^{-1} \mathbf{1}} \cdot \mathbf{1}
$$

This ensures the effect estimates $(\mu, \tau)$ are uncorrelated, making pattern classification ("is this shift or tail?") cleaner and single-effect MDE calculations accurate.

**Model limitations**: The linear tail basis assumes effects vary smoothly across quantiles. Real cache-timing attacks sometimes exhibit "hockey stick" patterns—flat through P80, then a sharp spike at P90. Such patterns will still be detected by the CI gate (which uses $\max|\Delta_p|$), but the Bayesian decomposition may attribute them to a mix of shift and tail rather than a pure tail effect. A model fit diagnostic (§2.8) flags when the observed $\Delta$ is poorly explained by the shift+tail model.

#### Prior Specification

We need a prior on $\beta$ under $H_1$. We use a zero-centered Gaussian:

$$
\beta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \text{diag}(\sigma_\mu^2, \sigma_\tau^2)
$$

The prior scale determines what we consider a "reasonably sized" effect. We set it adaptively based on what's detectable:

$$
\sigma_\mu = \max(2 \cdot \text{MDE}_\mu, \text{min\_effect\_of\_concern})
$$

$$
\sigma_\tau = \max(2 \cdot \text{MDE}_\tau, \text{min\_effect\_of\_concern})
$$

where MDE (minimum detectable effect) is computed from the calibration set's noise level. The factor of 2 ensures effects near the detection threshold get reasonable prior mass, while truly tiny effects (below min_effect_of_concern, default 10ns) are downweighted.

**Prior on hypotheses:** We default to $P(H_0) = 0.75$ (3:1 odds favoring no leak). This reflects that most well-written cryptographic code is constant-time. You can adjust this if you have different prior beliefs.

**Prior sensitivity**: The posterior probability depends on the prior. Users with different beliefs can use the reported Bayes factor to compute their own posteriors:

$$
\text{posterior odds} = \text{BF}_{10} \times \text{prior odds}
$$

#### Bayes Factor

The Bayes factor quantifies how much the data favor $H_1$ over $H_0$:

$$
\text{BF}_{10} = \frac{p(\Delta \mid H_1)}{p(\Delta \mid H_0)}
$$

A BF of 10 means the data are 10× more likely under "leak" than "no leak." A BF of 0.1 means the opposite.

Because both hypotheses are Gaussian, the marginal likelihoods have closed forms:

$$
\log \text{BF}_{10} = \log p(\Delta; 0, \Sigma_1) - \log p(\Delta; 0, \Sigma_0)
$$

where:
- $\Sigma_0$ is the null covariance (estimated via paired block bootstrap—see §2.6)
- $\Sigma_1 = \Sigma_0 + X \Lambda_0 X^\top$ is the alternative covariance, inflated by the prior on effects

The multivariate Gaussian log-PDF is:

$$
\log p(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = -\frac{1}{2} \left[ d \log(2\pi) + \log|\Sigma| + (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right]
$$

This requires computing the log-determinant and a matrix solve, both done via Cholesky decomposition.

#### Posterior Probability

The posterior probability of a leak combines the Bayes factor with the prior:

$$
P(\text{leak} \mid \Delta) = \frac{\text{BF}_{10} \cdot \text{prior\_odds}}{1 + \text{BF}_{10} \cdot \text{prior\_odds}}
$$

where $\text{prior\_odds} = P(H_1) / P(H_0) = 0.25 / 0.75 = 1/3$ by default.

This is just Bayes' rule. If the BF is 1 (equal evidence), the posterior equals the prior. Strong evidence shifts the posterior toward 0 or 1.

We clamp the result to $[0.0001, 0.9999]$ to prevent floating-point overflow when $|\log \text{BF}| > 700$.

#### Effect Estimation

When $P(\text{leak}) > 0.5$, we report an effect estimate. Under the $H_1$ model, the posterior on $\beta$ given $\Delta$ is also Gaussian (standard conjugate prior result⁴):

$$
\Lambda_{\text{post}} = \left( X^\top \Sigma_0^{-1} X + \Lambda_0^{-1} \right)^{-1}
$$

$$
\beta_{\text{post}} = \Lambda_{\text{post}} X^\top \Sigma_0^{-1} \Delta
$$

The posterior mean $\beta_{\text{post}} = (\mu, \tau)^\top$ gives the estimated shift and tail effects in nanoseconds.

⁴ Bishop, C. M. (2006). Pattern Recognition and Machine Learning, §3.3. Springer. These are the standard posterior mean and covariance for Bayesian linear regression with known observation noise.

**Effect pattern classification:**

Classification uses statistical significance rather than raw magnitudes. An effect component is "significant" if its magnitude exceeds twice its posterior standard error:

$$
\text{significant}(\mu) \iff |\mu| > 2 \cdot \sqrt{\Lambda_{\text{post}}[0,0]}
$$

| Shift significant? | Tail significant? | Pattern |
|--------------------|-------------------|---------|
| Yes | No | UniformShift |
| No | Yes | TailEffect |
| Yes | Yes | Mixed |
| No | No | Indeterminate |

When neither component is statistically significant, the pattern is reported as `Indeterminate` rather than guessing based on relative magnitudes. This indicates the effect is too small or noisy to characterize reliably.

**Credible interval:** To quantify uncertainty, we draw 500 samples from $\mathcal{N}(\beta_{\text{post}}, \Lambda_{\text{post}})$, compute the total effect magnitude $\lVert\beta\rVert_2$ for each, and report the 2.5th and 97.5th percentiles as a 95% credible interval.

### 2.6 Covariance Estimation

Both the CI gate and Bayesian layer need to know how much natural variability to expect in $\Delta$. This is captured by the null covariance $\Sigma_0$, estimated via paired block bootstrap on the calibration set.

**Why bootstrap instead of analytical formulas?**

Quantile estimators have complex covariance structures that depend on the unknown underlying density. Asymptotic formulas exist but require density estimation, which introduces its own errors. Bootstrap sidesteps this by directly resampling from the data.

**Why block bootstrap instead of standard bootstrap?**

Timing measurements are autocorrelated—nearby samples are more similar than distant ones due to cache state, branch predictor warmup, and frequency scaling. Standard bootstrap assumes i.i.d. samples; violating this leads to underestimated variance.

Block bootstrap preserves local correlation structure by resampling contiguous blocks rather than individual points. Block length scales as $n^{1/3}$ (Politis-Romano), with a constant that depends on the autocorrelation structure. We use:

$$
\hat{b} = \left\lceil c \cdot n^{1/3} \right\rceil, \quad c = 1.3 \text{ (default)}
$$

For $n = 30{,}000$ calibration samples, this gives blocks of ~40 samples. The constant $c$ is an engineering default; the theoretically optimal value depends on unknown spectral properties, but values in the range 1–2 work well empirically for timing data.³

³ Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." JASA 89(428):1303–1313. The $n^{1/3}$ scaling is well-established; the multiplicative constant requires spectral density estimation for optimality.

The block length constant $c$ is configurable for environments with unusual autocorrelation structure. If the pre-flight ACF check (§3.2) shows high autocorrelation persisting beyond the default block length, consider increasing $c$ to 2.0 or higher.

**Why joint resampling?**

With interleaved measurements, Fixed and Random samples measured near each other in time share environmental noise (cache state, CPU frequency, thermal conditions). This creates positive cross-covariance $\text{Cov}(\hat{q}_F, \hat{q}_R) > 0$, which *reduces* the variance of the difference:

$$
\text{Var}(\Delta) = \Sigma_F + \Sigma_R - 2\,\text{Cov}(\hat{q}_F, \hat{q}_R)
$$

To capture this benefit, we must resample the *joint interleaved sequence*, not the classes independently.

**Procedure:**

The calibration set is stored as an interleaved sequence of $(t_i, c_i)$ pairs, where $t_i$ is the timing and $c_i \in \{\text{Fixed}, \text{Random}\}$ is the class label.

For each of $B = 2{,}000$ bootstrap iterations:
1. Block-resample the **joint interleaved sequence** (preserving temporal structure)
2. **After resampling**, split by class label to obtain bootstrap samples for Fixed and Random
3. Compute quantile vectors $\hat{q}_F^*$ and $\hat{q}_R^*$ from the split samples
4. Compute $\Delta^* = \hat{q}_F^* - \hat{q}_R^*$

Compute sample covariance of the $\Delta^*$ vectors using Welford's online algorithm⁵ (numerically stable for large $B$).

⁵ Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. See also Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247 for the parallel/incremental generalization.

**Why this works:**

When we resample blocks from the joint sequence, samples that were measured together stay together. A block like:

$$
[(t_{100}, \text{Fixed}), (t_{101}, \text{Random}), (t_{102}, \text{Random}), (t_{103}, \text{Fixed}), \ldots]
$$

remains intact. The Fixed and Random samples within that block still share their environmental noise. After splitting, the bootstrap distribution of $\Delta^*$ correctly reflects the reduced variance from common-mode noise cancellation.

**Class-size fluctuation:**

After joint resampling, class sizes may fluctuate slightly (e.g., 10,050 Fixed / 9,950 Random instead of exactly 10,000 each). This $O(\sqrt{n})$ variation is negligible for quantile estimation and does not affect validity—both classes remain large enough for reliable estimates.

**Contrast with independent resampling:**

If we resampled Fixed and Random classes independently (block-resample each class separately), the bootstrap samples would lose their temporal pairing. Bootstrap iteration $k$'s Fixed samples would come from different original time indices than its Random samples, destroying the cross-covariance. This would yield $\text{Cov}(\hat{q}_F^*, \hat{q}_R^*) \approx 0$, overestimating variance by a factor that can approach 2× in high-common-mode-noise environments.

**Numerical stability:**

The covariance matrix must be positive definite for Cholesky decomposition. Near-singular matrices can arise from limited bootstrap iterations or degenerate timing distributions. Stability is defined by Cholesky success; if ill-conditioned:

1. Add jitter to the diagonal:

$$
\varepsilon = 10^{-10} + \frac{\text{tr}(\Sigma)}{9} \cdot 10^{-8}
$$

The trace-scaled term adapts to the matrix's magnitude.

2. If Cholesky still fails after jitter, return $\log \text{BF} = 0$ (neutral evidence). The posterior falls back to the prior, which is safe if uninformative.

**Variance floor:**

In idealized environments (simulators, deterministic operations), variance can approach zero, causing numerical instability. We apply a minimum variance floor based on timer quantization:

$$
\Sigma_0 \leftarrow \Sigma_0 + \frac{(\text{tick\_resolution})^2}{12} \cdot I
$$

The factor of $1/12$ is the variance of a uniform distribution over one tick—the irreducible quantization noise from discrete timers.

**Sample size scaling:**

The covariance $\Sigma_0$ is estimated from the calibration set ($n_{\text{cal}}$ samples per class), but applied to the inference set ($n_{\text{inf}}$ samples per class). Since quantile variance scales as $1/n$, we must scale the covariance to match the inference sample size:

$$
\Sigma_0 \leftarrow \Sigma_0 \cdot \frac{n_{\text{cal}}}{n_{\text{inf}}}
$$

With the default 30/70 split, this scales variance down by ~0.43 (= 30/70). Without this correction, the Bayesian layer would be systematically conservative—still valid, but with reduced statistical power and inflated MDE estimates.

**Edge case handling:**

If fewer than 2 bootstrap iterations complete (should never happen in practice), do not return an identity matrix—this would imply "1 ns² variance" which is arbitrary. Instead, return an error or a conservatively large diagonal (e.g., $10^6 \cdot I$) and flag the result as unreliable.

### 2.7 Minimum Detectable Effect

The MDE answers: "given the noise level in this measurement, what's the smallest effect I could reliably detect?"

This is important for interpreting negative results. If the MDE is 50ns and you're concerned about 10ns effects, a passing test doesn't mean your code is safe—it means your measurement wasn't sensitive enough. You'd need more samples or a quieter environment.

**Derivation:**

Under our linear model $\Delta = X\beta + \varepsilon$ with $\text{Var}(\varepsilon) = \Sigma_0$, the GLS estimator is:

$$
\hat{\beta} = (X^\top \Sigma_0^{-1} X)^{-1} X^\top \Sigma_0^{-1} \Delta
$$

with variance $\text{Var}(\hat{\beta}) = (X^\top \Sigma_0^{-1} X)^{-1}$.

For a single-effect model (estimating $\mu$ assuming $\tau = 0$, or vice versa), the variance of the projected estimator is:

$$
\text{Var}(\hat{\mu}) = \left( \mathbf{1}^\top \Sigma_0^{-1} \mathbf{1} \right)^{-1}, \quad
\text{Var}(\hat{\tau}) = \left( \mathbf{b}_{\text{tail}}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}
$$

The MDE is the effect size detectable at significance level $\alpha$ with 80% power:

$$
\text{MDE}_\mu = (z_{1-\alpha/2} + z_{0.8}) \cdot \sqrt{\left( \mathbf{1}^\top \Sigma_0^{-1} \mathbf{1} \right)^{-1}}
$$

$$
\text{MDE}_\tau = (z_{1-\alpha/2} + z_{0.8}) \cdot \sqrt{\left( \mathbf{b}_{\text{tail}}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}}
$$

where $z_{0.975} \approx 1.96$ for $\alpha = 0.05$ and $z_{0.8} \approx 0.84$, giving a combined factor of approximately 2.8.

**Intuition:**

The term $\mathbf{1}^\top \Sigma_0^{-1} \mathbf{1}$ is the *precision* of the uniform-shift estimator. Larger precision (smaller variance) means smaller MDE. In the simple case of i.i.d. quantiles with $\Sigma_0 = \sigma^2 I$:

$$
\text{MDE}_\mu \approx 2.8 \cdot \frac{\sigma}{\sqrt{9}} \approx 0.93\sigma
$$

Averaging 9 quantiles reduces the standard error by $\sqrt{9} = 3$, as expected.

**Interpretation:**

- MDE decreases with $\sqrt{n}$: 4× more samples → 2× better sensitivity
- MDE increases with timer noise: coarse timers mean larger MDE
- If MDE > min_effect_of_concern, consider more samples before trusting a "pass"

### 2.8 Diagnostics

Several diagnostics help assess result reliability and flag potential issues.

#### Non-Stationarity Check

The 30/70 temporal split assumes noise characteristics are stable across the measurement period. If a background process activates mid-measurement, the calibration covariance won't match inference conditions.

**Check:** Compare the empirical variance of the inference set (ignoring class labels) to the calibration set:

$$
R = \frac{\text{tr}(\hat{\Sigma}_{\text{inference}})}{\text{tr}(\hat{\Sigma}_{\text{calibration}})}
$$

| Ratio $R$ | Status |
|-----------|--------|
| 0.5–2.0 | Normal |
| 2.0–5.0 | Warning: environment may have changed |
| > 5.0 | Flag as non-stationary; results unreliable |

This is a rough heuristic. A variance increase of 2× might just be noise, but 5× strongly suggests something changed.

#### Model Fit Check

The Bayesian layer assumes $\Delta$ lies in the 2D subspace spanned by shift and tail effects. Real leaks may not fit this model (e.g., "hockey stick" patterns affecting only P90).

**Check:** After estimating $\hat{\beta}$, compute the residual:

$$
r = \Delta - X\hat{\beta}
$$

Under $H_1$ with correct model, $r \sim \mathcal{N}(0, \Sigma_0 - X\Lambda_{\text{post}}X^\top)$. Compute:

$$
\chi^2 = r^\top \Sigma_0^{-1} r
$$

This should be approximately $\chi^2_7$ (9 dimensions minus 2 parameters). If $\chi^2 > 18.5$ (p < 0.01), flag that the shift+tail model may not capture the leak pattern. The CI gate result remains valid; only the effect decomposition is suspect.

#### Outlier Asymmetry Check

Outlier filtering (§3.3) removes extreme values symmetrically. However, if one class has substantially more outliers than the other, that asymmetry may itself indicate a timing leak (one class has a heavy tail).

**Check:** After filtering, compare outlier rates:

$$
\text{rate}_F = \frac{\text{outliers trimmed from Fixed}}{\text{total Fixed samples}}
$$
$$
\text{rate}_R = \frac{\text{outliers trimmed from Random}}{\text{total Random samples}}
$$

| Condition | Action |
|-----------|--------|
| Both rates < 1% | Normal |
| One rate > 3× the other | Warning: asymmetric outliers may indicate tail leak |
| Absolute difference > 2% | Flag for investigation |

If outlier asymmetry is detected and the CI gate passes, consider re-running with a higher outlier threshold (99.99th percentile) or investigating the outlier distribution directly.

---

## 3. Measurement Model

This section describes how timing samples are collected: timer selection, warmup, interleaving, outlier handling, and adaptive batching for coarse-resolution platforms.

### 3.1 Timer Abstraction

Timing-oracle uses the highest-resolution timer available:

| Platform | Timer | Typical Resolution |
|----------|-------|-------------------|
| x86_64 | rdtsc | ~0.3 ns |
| x86_64 (with perf) | perf_event cycles | ~0.3 ns |
| Apple Silicon | cntvct_el0 | ~41 ns |
| Apple Silicon (with kperf) | PMU cycles | ~1 ns |
| Linux ARM64 | cntvct_el0 | ~40 ns |
| Linux ARM64 (with perf) | perf_event cycles | ~1 ns |

PMU-based timers (kperf, perf_event) require elevated privileges but provide much better resolution.

**Cycle-to-nanosecond conversion:**

Results are reported in nanoseconds for interpretability. The conversion factor is calibrated at startup by measuring a known delay.

### 3.2 Pre-flight Checks

Before measurement begins, several sanity checks detect common problems that would invalidate results:

**Timer sanity**: Verify the timer is monotonic and has reasonable resolution. Abort if the timer appears broken.

**Harness sanity (fixed-vs-fixed)**: Split the fixed samples in half and run the full analysis pipeline. If a "leak" is detected between two halves of identical inputs, something is wrong with the test harness—perhaps the closure captures mutable state, or the timer has systematic bias. This catches bugs that would otherwise produce false positives.

**Generator overhead**: Measure the baseline and sample generators in isolation. If they differ in cost by more than 10%, **abort with an error**. With the macro API, this check verifies that the harness is correctly pre-generating inputs (see §3.3)—if generation were accidentally happening inside the timed region, the measured difference would reflect generator cost rather than the operation under test. This check should always pass with correct harness implementation.

**Autocorrelation check**: Compute the autocorrelation function (ACF) on the full interleaved measurement sequence. If lag-1 or lag-2 autocorrelation exceeds 0.3, warn about periodic interference—likely from background processes, frequency scaling, or interrupt handlers. High autocorrelation inflates variance estimates and can cause both false positives and false negatives.

**CPU frequency governor (Linux)**: Check `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`. If not set to "performance", warn that frequency scaling may introduce noise. The "powersave" or "ondemand" governors cause the CPU to change frequency mid-measurement, adding variance unrelated to the code under test.

**Warmup**: Run the operation several times before measurement to warm caches, trigger JIT (if relevant), and stabilize frequency scaling. Default: 1,000 iterations.

### 3.3 Measurement Protocol

**Input pre-generation (critical):**

All inputs must be generated *before* the measurement loop begins. The timed region should contain only:
1. Input retrieval (loading a pre-generated value)
2. The operation under test
3. Output consumption (preventing dead-code elimination)

Input generation—especially random number generation—must happen outside the timed region. If random inputs are generated lazily inside the measurement closure, the generator overhead (often 20–100ns for cryptographic RNG) will be attributed to the operation, causing false positives.

```
// WRONG: Generator called inside timed region
for i in 0..n {
    let input = if schedule[i] == Fixed { &fixed } else { rng.gen() };  // ← RNG inside timing
    let t0 = timer.now();
    operation(input);
    let t1 = timer.now();
    record(t1 - t0);
}

// CORRECT: All inputs pre-generated
let fixed_inputs: Vec<_> = (0..n).map(|_| fixed.clone()).collect();
let random_inputs: Vec<_> = (0..n).map(|_| rng.gen()).collect();
for i in 0..n {
    let input = if schedule[i] == Fixed { &fixed_inputs[i] } else { &random_inputs[i] };
    let t0 = timer.now();
    operation(input);
    let t1 = timer.now();
    record(t1 - t0);
}
```

The "Generator overhead" pre-flight check (§3.2) verifies this requirement is met. With the macro API (§4.3), pre-generation is guaranteed by the harness—users cannot accidentally generate inputs inside the timed region.

**Interleaved randomized sampling:**

Rather than measuring all Fixed samples then all Random samples, we interleave them in randomized order:

1. Generate a schedule: [Fixed, Random, Random, Fixed, ...] with equal counts, shuffled
2. Execute according to schedule, recording timestamps
3. Separate results by class for analysis

This prevents systematic drift (thermal throttling, frequency scaling) from biasing one class.

**Outlier filtering:**

Extreme outliers (context switches, interrupts) can skew quantile estimates. Apply symmetric filtering:

1. Pool all samples from both classes
2. Compute the 99.9th percentile of the pooled distribution
3. Remove samples exceeding this threshold from both classes equally
4. **Track per-class outlier rates** (see §2.8 Outlier Asymmetry Check)

Report the fraction filtered (typically < 0.1%). High outlier rates indicate environmental noise.

**Caution:** Aggressive filtering can remove genuine signal. If outlier rates differ substantially between classes, the asymmetry diagnostic (§2.8) will flag this. Consider using a looser threshold (99.99th percentile) for operations with legitimately heavy tails.

### 3.4 Adaptive Batching

On platforms with coarse timers (cntvct at 41ns), fast operations may complete in fewer ticks than needed for reliable measurement.

**When batching is needed:**

If a pilot measurement shows fewer than 5 ticks per call, enable batching.

**Batch size selection:**

Choose $K$ such that $K$ operations take approximately 50 ticks:

$$
K = \text{clamp}\left( \left\lceil \frac{50}{\text{ticks\_per\_call}} \right\rceil, 1, 20 \right)
$$

The maximum of 20 prevents microarchitectural artifacts (branch predictor, cache state) from accumulating across too many iterations.

**Input symmetry within batches:**

When batching $K$ operations, each *measurement* (batch) must use the same input repeated $K$ times—for both classes. For the Random class, generate one random input per measurement, then repeat that input $K$ times within the batch. Do not use $K$ different random inputs within a single batch.

This ensures symmetric microarchitectural behavior: both classes see $K$ identical operations with identical cache/predictor training. Using $K$ different inputs for Random but one repeated input for Fixed would create artificial timing differences from branch predictor divergence, not from the operation under test.

**Effect scaling:**

When batching is enabled, inference is performed on the **sum** of $K$ operation timings. Only the **display** values (reported effect sizes) are divided by $K$. Do not divide by $K$ before quantile estimation—this would introduce floating-point smoothing that masks the discrete nature of timer ticks.

**Reporting:**

Results include batching metadata ($K$ value, achieved ticks, rationale) so users understand when batching was applied.

### 3.5 Measurability Thresholds

Some operations are too fast to measure reliably on a given platform.

**Thresholds:**

- If ticks per call $< 5$ (even with maximum batching): Return `Outcome::Unmeasurable`
- This typically means operations under ~200ns on Apple Silicon without kperf

**Unmeasurable result:**

Rather than returning unreliable statistics, the system explicitly indicates the limitation:

```rust
Outcome::Unmeasurable {
    operation_ns: 8.2,
    threshold_ns: 205.0,
    platform: "Apple Silicon (cntvct)",
    recommendation: "Run with sudo for kperf, or test a more complex operation"
}
```

### 3.6 Anomaly Detection

Users sometimes make mistakes that produce valid code but meaningless results—most commonly, capturing a pre-evaluated value instead of regenerating it:

```rust
let value = rand::random();  // Evaluated once!
timing_test! {
    baseline: || [0u8; 32],
    sample: || value,  // Always returns the same thing!
    measure: |input| compare(&input),
}
```

**Detection mechanism:**

During measurement, track the uniqueness of sample inputs by hashing the first 1,000 values generated. After measurement:

| Condition | Action |
|-----------|--------|
| All samples identical | Print error to stderr |
| $< 50\%$ unique | Print warning to stderr |
| Normal entropy | Silent |

**Type constraints:**

Anomaly detection requires the input type to be hashable. For types that don't support hashing (common in cryptography: scalars, field elements, big integers), detection is automatically skipped with zero overhead via autoref specialization.

### 3.7 Implementation Considerations

Several implementation details have statistical implications. Getting these wrong can introduce bias, inflate variance, or cause numerical instability.

#### Optimizer Barriers

Modern compilers aggressively optimize code, which can invalidate timing measurements in subtle ways:

- **Dead code elimination**: If the result of the operation isn't used, the compiler may remove it entirely
- **Code motion**: The compiler may move the operation outside the timed region
- **Constant folding**: With fixed inputs, the compiler may precompute results at compile time

The solution is `std::hint::black_box()`, which prevents the compiler from reasoning about a value's contents while having no runtime cost. Wrap both inputs and outputs:

```rust
let input = black_box(inputs.fixed());
let result = black_box(operation(&input));
```

For batched measurements, also use `compiler_fence(SeqCst)` between iterations to prevent the compiler from reordering or merging loop iterations. Without this, the compiler might transform a loop of K independent operations into something with different cache/predictor behavior.

**Statistical implication**: Without proper barriers, you may measure optimized-away code (artificially fast) or measure something other than what you intended. Both lead to invalid conclusions.

#### Quantile Computation

Quantiles are computed by sorting samples and interpolating. Two choices matter:

**Sorting algorithm**: Use unstable sort (`sort_unstable_by`) rather than stable sort. For timing data with many equal values (discrete ticks), stable sort's $O(n)$ extra memory and overhead is wasted. More importantly, benchmarking shows full sorting outperforms selection algorithms (like `select_nth_unstable`) for 9 quantiles due to cache behavior—the sorted array is reused 9 times.

**Comparison function**: Use `f64::total_cmp` rather than `partial_cmp`. While timing data shouldn't contain NaNs, using `partial_cmp(...).unwrap_or(Equal)` can silently corrupt sort ordering if NaNs appear (e.g., from division by zero in preprocessing). `total_cmp` provides a total ordering that handles NaNs consistently.

**Interpolation method**: Use R-7 linear interpolation (the default in R and NumPy). For sorted sample $x$ of size $n$ at probability $p$:

$$
h = (n-1) \cdot p, \quad \hat{q}_p = x_{\lfloor h \rfloor} + (h - \lfloor h \rfloor)(x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor})
$$

R-7 produces smoother quantile estimates than direct order statistics, which matters for the calibration set (smaller $n$). For $n > 1000$, the difference is negligible, but consistency across sample sizes avoids subtle biases.

**Statistical implication**: The choice of interpolation method affects quantile estimates at small sample sizes. Using a consistent, well-understood method (R-7) ensures our covariance estimates and thresholds are calibrated correctly.

#### Numerical Stability in Covariance Estimation

The covariance matrix $\Sigma_0$ is estimated from 2,000 bootstrap samples, each a 9-dimensional quantile vector. Naive computation (accumulate sum, then divide) suffers from catastrophic cancellation when the mean is large relative to the variance.

Use Welford's online algorithm, which maintains a running mean and sum of squared deviations:

```
for each sample x:
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)  # Note: uses updated mean
covariance = M2 / (n - 1)
```

This is numerically stable regardless of the mean's magnitude. For covariance matrices, extend to track cross-products.

**Statistical implication**: Numerical instability in covariance estimation propagates to the Bayes factor calculation. A poorly estimated $\Sigma_0$ can cause the Cholesky decomposition to fail or produce incorrect posteriors. Welford's algorithm avoids this failure mode.

#### Batching and Microarchitectural Artifacts

When batching K operations together (§3.4), microarchitectural state accumulates across iterations. This can cause timing differences between classes even for constant-time code:

- **Branch predictor training**: K iterations with identical input trains predictors differently than K varied inputs
- **Cache state**: Same cache lines accessed K times vs K different access patterns
- **μop cache**: Same instruction sequence cached and optimized vs varied sequences

These are measurement artifacts, not timing leaks. We limit $K \leq 20$ to bound these effects—empirically, K=20 keeps artifacts below the noise floor for typical crypto operations.

Additional mitigations:

- Keep K identical across both classes (never batch Fixed at K=10 and Random at K=15)
- Pre-generate all random inputs before the timed region (generator cost doesn't confound measurement)
- Perform inference on batch totals, not per-call divided values—division reintroduces quantization noise

**Statistical implication**: Without these mitigations, batching can produce false positives (detecting "leaks" that are actually predictor/cache artifacts) or false negatives (artifacts masking real leaks). The K=20 limit and mitigations keep the false positive rate within the advertised $\alpha$.

---

## 4. Public API

This section describes the user-facing types and functions: result structures, configuration options, and three API tiers (macro, builder, raw) for different use cases.

### 4.1 Result Types

#### TestResult

The primary result type, returned directly by `timing_test!`:

```rust
pub struct TestResult {
    /// Posterior probability of timing leak (0.0 to 1.0)
    pub leak_probability: f64,
    
    /// Bayes factor (H1 vs H0) for users who want to apply different priors
    pub bayes_factor: f64,
    
    /// Effect size estimate (when leak_probability > 0.5)
    pub effect: Option<Effect>,
    
    /// Exploitability assessment
    pub exploitability: Exploitability,
    
    /// Minimum detectable effect given noise level (at 80% power)
    pub min_detectable_effect: MinDetectableEffect,
    
    /// CI gate result
    pub ci_gate: CiGate,
    
    /// Measurement quality assessment
    pub quality: MeasurementQuality,
    
    /// Fraction of samples trimmed as outliers
    pub outlier_fraction: f64,
    
    /// Diagnostics (non-stationarity, model fit, outlier asymmetry)
    pub diagnostics: Diagnostics,
}
```

#### Outcome

Returned by `timing_test_checked!` and the builder API for explicit unmeasurable handling:

```rust
pub enum Outcome {
    /// Analysis completed successfully
    Completed(TestResult),
    
    /// Operation too fast to measure on this platform
    Unmeasurable {
        operation_ns: f64,
        threshold_ns: f64,
        platform: String,
        recommendation: String,
    },
}
```

Most users should use `timing_test!` which panics on unmeasurable operations. Use `timing_test_checked!` or the builder API when you need to handle unmeasurable cases gracefully (e.g., conditional test skipping).

#### Effect

When a leak is detected, the effect is decomposed:

```rust
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = baseline class slower)
    pub shift_ns: f64,
    
    /// Tail effect in nanoseconds (positive = baseline has heavier upper tail)
    pub tail_ns: f64,
    
    /// 95% credible interval for total effect magnitude
    pub credible_interval_ns: (f64, f64),
    
    /// Dominant pattern
    pub pattern: EffectPattern,
}

pub enum EffectPattern {
    UniformShift,   // Branch, different code path
    TailEffect,     // Cache misses, memory access patterns
    Mixed,          // Both components significant (check model_fit diagnostic for non-linear patterns)
    Indeterminate,  // Neither component statistically significant
}
```

#### CiGate

The pass/fail decision with controlled FPR:

```rust
pub struct CiGate {
    pub alpha: f64,             // Target FPR
    pub passed: bool,           // True if no leak detected
    pub threshold: f64,         // Effective threshold (after flooring at 1 tick and min_effect_of_concern)
    pub max_observed: f64,      // Observed max|Δ_p|
    pub observed: [f64; 9],     // Per-quantile differences (for diagnostics)
}
```

#### Diagnostics

Flags from §2.8 diagnostic checks:

```rust
pub struct Diagnostics {
    /// Non-stationarity: ratio of inference to calibration variance
    pub stationarity_ratio: f64,
    pub stationarity_ok: bool,
    
    /// Model fit: chi-squared statistic for residuals
    pub model_fit_chi2: f64,
    pub model_fit_ok: bool,
    
    /// Outlier asymmetry: per-class outlier rates
    pub outlier_rate_fixed: f64,
    pub outlier_rate_random: f64,
    pub outlier_asymmetry_ok: bool,
    
    /// Human-readable warnings (empty if all OK)
    pub warnings: Vec<String>,
}
```

#### Exploitability

Heuristic assessment based on Crosby et al. (2009):

```rust
pub enum Exploitability {
    Negligible,      // < 100 ns: at LAN resolution limit; requires ~10k+ queries
    PossibleLAN,     // 100–500 ns: exploitable on LAN with ~1k–10k queries
    LikelyLAN,       // 500 ns – 20 μs: readily exploitable on LAN
    PossibleRemote,  // > 20 μs: potentially exploitable over internet
}
```

**Note:** These thresholds are based on 2009 measurements and are approximate. Modern networks and attack techniques may achieve better resolution. Treat these as rough guidance rather than definitive boundaries.

#### MeasurementQuality

Assessment of result reliability:

```rust
pub enum MeasurementQuality {
    /// Confident in results
    Good,
    /// Results valid but noisier than ideal
    Acceptable { issues: Vec<String> },
    /// Results may be unreliable
    TooNoisy { issues: Vec<String> },
}
```

### 4.2 Configuration

#### TimingOracle

The main configuration type with builder pattern:

```rust
impl TimingOracle {
    pub fn new() -> Self;       // 100k samples, thorough
    pub fn balanced() -> Self;  // 20k samples, good for CI
    pub fn quick() -> Self;     // 5k samples, development
    
    pub fn samples(self, n: usize) -> Self;
    pub fn warmup(self, n: usize) -> Self;
    pub fn ci_alpha(self, alpha: f64) -> Self;
    pub fn min_effect_of_concern(self, ns: f64) -> Self;
}
```

#### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| samples | 100,000 | Samples per class |
| warmup | 1,000 | Warmup iterations |
| ci_alpha | 0.01 | CI gate false positive rate ($\alpha$) |
| min_effect_of_concern | 10.0 ns | Effects below this are considered negligible |
| prior_no_leak | 0.75 | Prior probability of no leak ($P(H_0)$) |

### 4.3 Macro API

The `timing_test!` macro is the recommended API for most users. It returns `TestResult` directly and panics if the operation is unmeasurable.

#### Syntax

```rust
// Setup: define any variables needed for the test
let key = Aes128::new(&KEY);

let result = timing_test! {
    // Optional: custom configuration
    oracle: TimingOracle::balanced(),
    
    // Optional: explicit input type (improves error messages)
    type Input = [u8; 32],
    
    // Required: baseline input (closure called once per measurement)
    baseline: || [0u8; 32],
    
    // Required: sample generator (closure called once per measurement)
    sample: || rand::random::<[u8; 32]>(),
    
    // Required: operation to measure
    measure: |input| {
        key.encrypt(&input);
    },
};

// result is TestResult, not Outcome
assert!(result.ci_gate.passed);
```

Variables defined before the macro are captured by the closures using normal Rust closure semantics—no magic scoping rules.

**Pre-generation contract:**

The `baseline` and `sample` closures are used to **pre-generate** all inputs before any timing measurements begin. They are:

- Called in batch to fill input buffers (not interleaved with timing)
- Never invoked inside the timed region
- Completely finished before the first `measure` call

This separation prevents input generation from contaminating timing via cache pollution, branch predictor warmup, allocator noise, or thermal effects. The `measure` closure receives pre-generated inputs and is the *only* code inside the timed region.

**Both `baseline` and `sample` are closures.** This symmetry makes the API consistent. The difference is semantic: `baseline` typically returns a fixed value (the "control"), while `sample` generates varied inputs.

#### Checked Variant

For explicit unmeasurable handling, use `timing_test_checked!`:

```rust
let outcome = timing_test_checked! {
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| operation(&input),
};

match outcome {
    Outcome::Completed(result) => assert!(result.ci_gate.passed),
    Outcome::Unmeasurable { recommendation, .. } => {
        eprintln!("Skipping: {}", recommendation);
    }
}
```

Use `timing_test_checked!` when:
- Running on platforms with coarse timers where unmeasurable is expected
- You want to skip rather than fail when measurement isn't possible
- Writing platform-adaptive test suites

#### Multiple Inputs

Use tuple destructuring:

```rust
let cipher = ChaCha20Poly1305::new(&key);

timing_test! {
    type Input = ([u8; 12], [u8; 64]),
    
    baseline: || ([0u8; 12], [0u8; 64]),
    sample: || (rand::random(), rand::random()),
    measure: |(nonce, plaintext)| {
        cipher.encrypt(&nonce, &plaintext);
    },
}
```

#### Compile-Time Errors

The macro catches common mistakes with clear messages:

| Mistake | Error |
|---------|-------|
| Missing `sample:` field | "missing `sample` field" |
| Typo in field name | "unknown field `sampel`, expected one of: baseline, sample, measure, ..." |
| Type mismatch baseline/sample | Standard Rust type error pointing at user's code |
| Missing type annotation when needed | Suggest adding `type Input = ...` |

### 4.4 Builder API

For users who prefer explicit code over macros:

```rust
let result = TimingTest::new()
    .oracle(TimingOracle::balanced())
    .baseline(|| [0u8; 32])
    .sample(|| rand::random::<[u8; 32]>())
    .measure(|input| secret.ct_eq(&input))
    .run();
```

#### Methods

```rust
impl TimingTest {
    pub fn new() -> Self;
    pub fn oracle(self, oracle: TimingOracle) -> Self;      // Optional
    pub fn baseline<F: FnMut() -> V>(self, f: F) -> Self;   // Required
    pub fn sample<F: FnMut() -> V>(self, gen: F) -> Self;   // Required
    pub fn measure<E: FnMut(V)>(self, body: E) -> Self;     // Required
    pub fn run(self) -> Outcome;
}
```

#### Error Handling

| Condition | Behavior |
|-----------|----------|
| Missing required field | Panic with clear message |
| Type mismatch | Compile error |
| Unmeasurable operation | Returns `Outcome::Unmeasurable` |

### 4.5 Raw API

For complex cases requiring full control:

```rust
use timing_oracle::{measure, helpers::InputPair};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());

let result = measure(
    || encrypt(&inputs.baseline()),
    || encrypt(&inputs.sample()),
);
```

#### InputPair

Separates input generation from measurement:

```rust
impl<F1: FnMut() -> T, F2: FnMut() -> T> InputPair<T, F1, F2> {
    pub fn new(baseline: F1, sample: F2) -> Self;
    pub fn baseline(&mut self) -> T;  // Calls baseline closure
    pub fn sample(&mut self) -> T;    // Calls sample closure
}
```

**Anomaly detection**: If the type is hashable, tracks value uniqueness and warns about suspicious patterns. For non-hashable types, tracking is skipped automatically with zero overhead.

#### When to Use Each API

| API | Returns | Best for |
|-----|---------|----------|
| `timing_test!` | `TestResult` | Most users; panics if unmeasurable |
| `timing_test_checked!` | `Outcome` | Platform-adaptive tests; explicit unmeasurable handling |
| `TimingTest` builder | `Outcome` | Macro-averse; programmatic result access |
| `InputPair` + `measure()` | `Outcome` | Multiple independent varying inputs; full control |

---

## 5. Interpreting Results

### 5.1 Leak Probability and Bayes Factor

The `leak_probability` is a Bayesian posterior: the probability that there is a timing leak given the data and model assumptions.

**Decision thresholds:**

| Probability | Interpretation | Action |
|-------------|----------------|--------|
| $< 10\%$ | Probably safe | Pass |
| $10\%$–$50\%$ | Inconclusive | Consider more samples |
| $50\%$–$90\%$ | Probably leaking | Investigate |
| $> 90\%$ | Almost certainly leaking | Fix required |

These are guidelines, not rules. Your risk tolerance may vary.

**Using the Bayes factor:**

The `bayes_factor` field gives the raw evidence ratio, independent of the prior. Users with different prior beliefs can compute their own posteriors:

$$
\text{posterior odds} = \text{BF}_{10} \times \frac{P(H_1)}{P(H_0)}
$$

$$
P(\text{leak}) = \frac{\text{posterior odds}}{1 + \text{posterior odds}}
$$

For example, if BF₁₀ = 5 and you believe $P(H_0) = 0.9$ (9:1 odds favoring no leak):
- Prior odds = 1/9 ≈ 0.11
- Posterior odds = 5 × 0.11 ≈ 0.56
- P(leak) ≈ 36%

**What it is not:**

- Not a confidence level
- Not a p-value
- Not the probability of being hacked

It's the probability that timing depends on input, given reasonable priors about effect sizes.

### 5.2 CI Gate

The CI gate provides a binary pass/fail with controlled false positive rate.

**Semantics:**

- `passed: true` means the maximum quantile difference did not exceed the permutation threshold at the $\alpha$ level
- `passed: false` means the maximum quantile difference exceeded the threshold

**Relationship to Bayesian layer:**

The gate and posterior usually agree, but can diverge:

| CI Gate | Posterior | Interpretation |
|---------|-----------|----------------|
| Pass | Low | Clean bill of health |
| Pass | Medium | Small effect below detection threshold |
| Fail | High | Clear leak |
| Fail | Medium | Detected but effect is small |

When they diverge, trust the gate for CI decisions (it controls FPR) and use the posterior for understanding magnitude.

### 5.3 Effect Size

When a leak is detected, the effect tells you *how big* and *what kind*:

**Shift ($\mu$)**: Uniform timing difference across all quantiles. Typical cause: different code path, branch on secret data.

**Tail ($\tau$)**: Upper quantiles affected more than lower. Typical cause: cache misses, memory access patterns depending on secret.

**Interpreting magnitude:**

The effect is in nanoseconds. A 50ns shift means the fixed input takes about 50ns longer on average than random inputs.

**Credible interval:**

The 95% credible interval gives uncertainty bounds. If it's [20ns, 80ns], you're 95% confident the true effect is in that range.

**Indeterminate pattern:**

When neither shift nor tail is statistically significant (both within 2 standard errors of zero), the pattern is `Indeterminate`. This typically means:
- The effect is very small (near the noise floor)
- The effect has an unusual shape not captured by shift+tail
- Check the model fit diagnostic for the latter case

**Mixed pattern:**

When both shift and tail are significant, the pattern is `Mixed`. This can indicate either:
- A genuine combination of both effects (e.g., a code path that's both slower overall and has more variable cache behavior)
- A non-linear pattern (e.g., "hockey stick" affecting only P90) that the linear model approximates as shift+tail

If `diagnostics.model_fit_ok` is false alongside a `Mixed` pattern, inspect `ci_gate.observed` for the raw quantile differences—the true pattern may not decompose cleanly into shift and tail components.

### 5.4 Exploitability

The `Exploitability` enum provides a rough assessment of practical risk, based on Crosby et al. (2009)⁶ which measured timing attack resolution across network conditions:

| Level | Effect Size | Meaning |
|-------|-------------|---------|
| Negligible | $< 100$ ns | At the limit of LAN exploitability; requires ~10k+ queries |
| PossibleLAN | $100$–$500$ ns | Exploitable on LAN with ~1k–10k queries |
| LikelyLAN | $500$ ns – $20$ μs | Readily exploitable on LAN with hundreds of queries |
| PossibleRemote | $> 20$ μs | Large enough to potentially exploit over internet |

⁶ Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. Key findings: LAN resolution "as good as 100ns" with thousands of measurements; internet resolution "15–100µs" with best hosts resolving ~30µs. Their empirical tests with 1,000 measurements showed ~200ns resolution on LAN and ~30–50µs over the internet.

**Caveats:**

- These are heuristics based on 2009 measurements, not guarantees
- Modern networks and attacks may achieve better resolution
- Actual exploitability depends on many factors (network jitter, attacker capabilities, protocol specifics)
- Even "Negligible" leaks should be fixed if practical
- **Amplification attacks**: These thresholds assume single-query attacks. Amplification attacks (e.g., Lucky13⁷) can exploit smaller timing differences if the attacker can trigger them repeatedly within a protocol. A 100ns leak in a byte-by-byte comparison loop, queried millions of times, becomes exploitable even at "Negligible" magnitude.

⁷ AlFardan, N. J. & Paterson, K. G. (2013). "Lucky Thirteen: Breaking the TLS and DTLS Record Protocols." IEEE S&P. Exploited timing differences around 1μs using ~2²³ queries.

### 5.5 Quality Assessment

The `MeasurementQuality` indicates confidence in the results, primarily based on the minimum detectable effect (MDE) relative to effects of practical concern:

| MDE | Quality | Interpretation |
|-----|---------|----------------|
| $< 5$ ns | Excellent | Can detect very small leaks |
| $5$–$20$ ns | Good | Sufficient for most applications |
| $20$–$100$ ns | Acceptable | May miss small leaks; consider more samples |
| $> 100$ ns | TooNoisy | Results unreliable; environment too noisy |

These thresholds assume a default `min_effect_of_concern` of 10ns. If you're concerned about smaller effects, you need correspondingly better quality.

**Other quality factors:**

**Good**: Normal conditions, MDE within acceptable range, results are reliable.

**Acceptable**: Minor issues detected but results are still valid:
- Moderately high outlier rate (1–5%)
- MDE somewhat elevated but still useful
- Slight autocorrelation in measurements

**TooNoisy**: Results may be unreliable:
- Very high outlier rate ($> 5\%$)
- MDE exceeds 100ns (can't detect practically relevant effects)
- Severe autocorrelation
- Timer resolution insufficient even with batching

When quality is `TooNoisy`, consider:
- Reducing system load (close browsers, stop background jobs)
- Increasing sample count ($4\times$ samples → $2\times$ better MDE)
- Running on dedicated hardware or in single-user mode
- Using a platform with better timer resolution (x86_64 with rdtsc, or ARM with perf_event)

### 5.6 Diagnostics

Check the `diagnostics` field for potential issues:

**Non-stationarity** (`stationarity_ok: false`): The noise level changed between calibration and inference. Results may be unreliable. Consider re-running when the system is more stable.

**Model fit** (`model_fit_ok: false`): The observed quantile differences don't fit the shift+tail model well. The CI gate result is still valid, but the effect decomposition may be misleading. Inspect `ci_gate.observed` for the raw quantile differences.

**Outlier asymmetry** (`outlier_asymmetry_ok: false`): One class had substantially more outliers than the other. This asymmetry may itself indicate a timing leak (heavy tail in one class). Consider re-running with a higher outlier threshold.

### 5.7 Reliability Handling

Some tests may be unreliable on certain platforms due to timer limitations or environmental factors.

**Checking reliability:**

```rust
impl Outcome {
    /// True if results are trustworthy
    pub fn is_reliable(&self) -> bool;
}
```

A result is reliable if:
- Measurement completed (not `Unmeasurable`)
- Quality is not `TooNoisy`, OR posterior is conclusive ($< 0.1$ or $> 0.9$)
- No critical diagnostic failures

The rationale: conclusive posteriors overcame the noise, so the signal was strong enough.

**Handling unreliable results:**

For CI integration, use fail-open or fail-closed policies:

```rust
// Fail-open: unreliable tests are skipped (pass)
let result = skip_if_unreliable!(outcome, "test_name");

// Fail-closed: unreliable tests fail
let result = require_reliable!(outcome, "test_name");
```

Environment variable `TIMING_ORACLE_UNRELIABLE_POLICY` can override: set to `fail_closed` for stricter CI.

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\Delta$ | 9-dimensional vector of signed quantile differences |
| $\Delta_p$ | Quantile difference at percentile $p$ |
| $\Sigma_0$ | Null covariance matrix of $\Delta$ (estimated via paired bootstrap) |
| $\Sigma_1$ | Alternative covariance (leak hypothesis) |
| $X$ | $9 \times 2$ design matrix $[\mathbf{1} \mid \mathbf{b}_{\text{tail}}]$ |
| $\beta = (\mu, \tau)^\top$ | Effect parameters: shift and tail |
| $\Lambda_0$ | Prior covariance for $\beta$ |
| $\text{BF}_{10}$ | Bayes factor for $H_1$ vs $H_0$ |
| $\alpha$ | CI gate false positive rate |
| MDE | Minimum detectable effect (at 80% power) |
| $K$ | Batch size for adaptive batching |
| $n$ | Samples per class |
| $B$ | Bootstrap/permutation iterations |

## Appendix B: Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| Deciles | $\{0.1, 0.2, \ldots, 0.9\}$ | Nine quantile positions |
| CI permutation iterations | $10{,}000$ | Sufficient for stable threshold estimation |
| Covariance bootstrap iterations | $2{,}000$ | Balance between accuracy and speed |
| Min samples per class | $200$ | Below this, extreme quantile estimation is unreliable |
| Min ticks per call | $5$ | Below this, quantization noise dominates |
| Target ticks per batch | $50$ | Target for adaptive batching |
| Max batch size | $20$ | Limit microarchitectural artifacts |
| Anomaly detection window | $1{,}000$ | Samples to track for uniqueness |
| Anomaly detection threshold | $0.5$ | Warn if $< 50\%$ unique values |
| Default prior $P(H_0)$ | $0.75$ | Conservative prior favoring no leak |
| Default $\alpha$ | $0.01$ | 1% false positive rate |
| Default min effect | $10$ ns | Effects below this are negligible |

## Appendix C: References

**Statistical methodology:**

1. Dunsche, M., Lamp, M., & Pöpper, C. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF bootstrap methodology

2. Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics. — Block bootstrap for autocorrelated data

3. Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." JASA 89(428):1303–1313. — Block length heuristics

4. Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. — R-7 quantile interpolation

5. Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 3. Springer. — Bayesian linear regression

6. Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. — Online variance algorithm

7. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247. — Parallel Welford extension

**Timing attacks:**

8. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — DudeCT methodology

9. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. — Exploitability thresholds. Key findings: LAN resolution ~100ns with thousands of measurements; internet ~15–100µs.

10. AlFardan, N. J. & Paterson, K. G. (2013). "Lucky Thirteen: Breaking the TLS and DTLS Record Protocols." IEEE S&P. — Amplification attacks exploiting ~1μs timing differences.

**Existing tools:**

11. dudect (C): https://github.com/oreparaz/dudect
12. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher
