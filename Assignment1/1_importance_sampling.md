# 1. Importance Sampling

---

## 1.1 The Target Function

The function we want to integrate is:

$$f(x) = x^{1/3} e^{-x^2}$$

and we seek the value of the definite integral:

$$I = \int_{0}^{2} f(x) \, dx$$

---

## 1.2 Standard Monte Carlo Estimator

For $X \sim \text{Unif}(0, 2)$, the probability density is $p(x) = \frac{1}{b-a} = \frac{1}{2}$. Rewriting the integral as an expectation under this uniform distribution:

$$I = \int_{0}^{2} f(x) \, dx = (b - a) \cdot \mathbb{E}[f(X)] = 2 \cdot \mathbb{E}\left[X^{1/3} e^{-X^2}\right]$$

This motivates the standard Monte Carlo estimator:

$$\hat{I}_{MC} = \frac{2}{n} \sum_{i=1}^{n} f(X_i) = \frac{2}{n} \sum_{i=1}^{n} X_i^{1/3} e^{-X_i^2}, \quad X_i \overset{iid}{\sim} \text{Unif}(0,2)$$

---

## 1.3 Importance Sampling

### (a) Determining the Normalising Constant $c$

We consider the proposal density $q(x) = c\, x^{1/3}$ on $[0, 2]$. For $q$ to be a valid PDF, it must integrate to 1:

$$\int_{0}^{2} c\, x^{1/3} \, dx = 1$$

Evaluating the integral:

$$c \left[\frac{x^{4/3}}{4/3}\right]_0^2 = c \cdot \frac{3}{4} \cdot 2^{4/3} = 1$$

Solving for $c$:

$$\boxed{c = \frac{4}{3 \cdot 2^{4/3}}}$$

### (b) Inverse Transform Sampling

To draw samples $X_i \sim q$, we apply the inverse CDF method.

**Step 1 — Compute the CDF $Q(x)$:**

$$Q(x) = \int_{0}^{x} c\, t^{1/3} \, dt = c \cdot \frac{3}{4} x^{4/3}$$

Substituting $c = \frac{4}{3 \cdot 2^{4/3}}$:

$$Q(x) = \frac{x^{4/3}}{2^{4/3}} = \left(\frac{x}{2}\right)^{4/3}$$

**Step 2 — Invert the CDF:**

Setting $Q(x) = U$ for $U \sim \text{Unif}(0, 1)$ and solving for $x$:

$$U = \left(\frac{x}{2}\right)^{4/3} \implies x = 2U^{3/4}$$

Samples are therefore generated as $X_i = 2U_i^{3/4}$ where $U_i \overset{iid}{\sim} \text{Unif}(0,1)$.

### (c) The Importance Sampling Estimator

The general importance sampling estimator for $I = \int f(x)\,dx$ using proposal $q$ is:

$$\hat{I}_{IS} = \frac{1}{n} \sum_{i=1}^{n} \frac{f(X_i)}{q(X_i)}, \quad X_i \overset{iid}{\sim} q$$

Substituting $f(x) = x^{1/3} e^{-x^2}$ and $q(x) = c\, x^{1/3}$, the $x^{1/3}$ terms cancel:

$$\hat{I}_{IS} = \frac{1}{n} \sum_{i=1}^{n} \frac{X_i^{1/3} e^{-X_i^2}}{c\, X_i^{1/3}} = \frac{1}{cn} \sum_{i=1}^{n} e^{-X_i^2}$$

---

## 1.4 Discussion

**Magnitude of variance reduction.** In this experiment, importance sampling did not yield a meaningful reduction in variance. The standard deviations of both estimators were similar, and the importance sampling estimator occasionally exhibited slightly *higher* variance than standard Monte Carlo. This is because the proposal $q$ does not minimise the variance of the importance weights across the full interval — a point discussed further below.

**Stability and concentration.** Despite the absence of variance reduction, both estimators are highly stable. With $n = 10^6$, the Law of Large Numbers ensures standard deviations on the order of $5 \times 10^{-4}$, so both methods reliably estimate $I$ to several decimal places.

**Computational cost.** The importance sampling estimator is slightly cheaper per iteration. Because the $x^{1/3}$ factor cancels analytically in the ratio $f/q$, each evaluation reduces to a single exponential $e^{-x^2}$. The standard Monte Carlo estimator must compute both a fractional power $x^{1/3}$ and an exponential for every sample, requiring more floating-point operations.

**Suitability of $q(x) = c\, x^{1/3}$.** The proposal is only partially suitable. Near $x = 0$, it correctly mirrors the power-law growth of $f$. However, as $x \to 2$, the integrand $f(x)$ decays sharply to zero due to the $e^{-x^2}$ factor, while $q(x)$ continues to grow. This mismatch causes the sampler to over-weight the right end of the interval where $f$ contributes very little, which inflates the variance of the importance weights rather than reducing it.

**How the shape of $f$ motivates good proposal choices.** An ideal proposal $q$ should track the shape of $|f(x)|$ closely, keeping the ratio $f(x)/q(x)$ approximately constant. Inspecting the plot of $f$, a well-designed $q$ would need to capture both the initial power-law rise near zero and the rapid exponential decay toward $x = 2$. A truncated distribution proportional to $x^{1/3} e^{-x^2}$ itself would be perfect but intractable; a practical alternative might use a Gaussian or gamma-shaped envelope that falls off in the same region as $f$.
