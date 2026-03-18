# 3. Metropolis-Hastings Algorithm

The observed data are $x = (4, 1, 3, 5, 2, 4)$, giving $n = 6$ and $\sum_{i=1}^n x_i = 19$. The prior is $\text{Gamma}(\alpha, \beta)$ with $\alpha = \beta = 1$.

---

## 3.1 Posterior Density Derivation

By Bayes' theorem, the posterior is proportional to the likelihood times the prior:

$$p(\lambda \mid x) \propto p(x \mid \lambda)\, \pi(\lambda)$$

For $n$ independent $\text{Poisson}(\lambda)$ observations, the joint likelihood is:

$$p(x \mid \lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} \propto \lambda^{\sum x_i} e^{-n\lambda}$$

The $\text{Gamma}(\alpha, \beta)$ prior contributes:

$$\pi(\lambda) \propto \lambda^{\alpha - 1} e^{-\beta\lambda}$$

Multiplying and collecting terms:

$$p(\lambda \mid x) \propto \lambda^{\alpha - 1 + \sum x_i} \exp\!\bigl(-(\beta + n)\lambda\bigr)$$

This is the kernel of a $\text{Gamma}(\alpha + \sum x_i,\; \beta + n)$ distribution. With our data:

$$\boxed{p(\lambda \mid x) \sim \text{Gamma}(20,\; 7)}$$

with posterior mean $\mathbb{E}[\lambda \mid x] = 20/7 \approx 2.857$ and posterior variance $20/49 \approx 0.408$.

---

## 3.2 Random Walk Metropolis-Hastings

### (a) Acceptance Probability

The general MH acceptance probability is:

$$A(\lambda, \lambda') = \min\!\left(1,\; \frac{p(\lambda' \mid x)}{p(\lambda \mid x)} \cdot \frac{q(\lambda \mid \lambda')}{q(\lambda' \mid \lambda)}\right)$$

The log-normal proposal $\lambda' = \lambda \exp(\epsilon)$, $\epsilon \sim N(0, \sigma^2)$, is not symmetric on $\mathbb{R}^+$, so the proposal ratio does not cancel. Applying the change of variables $\epsilon = \log(\lambda'/\lambda)$ with Jacobian $|d\epsilon/d\lambda'| = 1/\lambda'$, the proposal density is:

$$q(\lambda' \mid \lambda) = \frac{1}{\lambda'\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(\log\lambda' - \log\lambda)^2}{2\sigma^2}\right)$$

In the ratio $q(\lambda \mid \lambda') / q(\lambda' \mid \lambda)$, the Gaussian exponentials are identical (the argument is symmetric in $\lambda$ and $\lambda'$) and the normalising constants cancel, leaving only the Jacobian terms:

$$\frac{q(\lambda \mid \lambda')}{q(\lambda' \mid \lambda)} = \frac{1/\lambda}{1/\lambda'} = \frac{\lambda'}{\lambda}$$

Substituting into the acceptance probability and collecting powers of $\lambda'/\lambda$:

$$A(\lambda, \lambda') = \min\!\left(1,\; \left(\frac{\lambda'}{\lambda}\right)^{\!\alpha + \sum x_i} \exp\!\bigl(-(\beta + n)(\lambda' - \lambda)\bigr)\right)$$

With $\alpha + \sum x_i = 20$ and $\beta + n = 7$ this becomes:

$$A(\lambda, \lambda') = \min\!\left(1,\; \left(\frac{\lambda'}{\lambda}\right)^{\!20} \exp\!\bigl(-7(\lambda' - \lambda)\bigr)\right)$$

### (b) Choosing $\sigma^2$ by Trial

There is a fundamental tension in choosing $\sigma^2$: a small variance produces proposals close to the current state, which are nearly always accepted but explore the posterior slowly; a large variance generates bold proposals that are frequently rejected, stalling the chain. The conventional target acceptance rate for a univariate random walk is between 25% and 40%.

The script evaluates all variances in the grid and selects the one whose acceptance rate falls closest to the midpoint of the target window (32.5%). The selected value was $\sigma^2 = 0.65$, yielding an acceptance rate of approximately 33%.

Note that the pilot and final chains use different random draws, so the reported rates can differ slightly; the selected $\sigma^2$ is a heuristic starting point rather than a guarantee.

---

## 3.3 Independence Sampler

### (a) Acceptance Ratio Derivation

In an independence sampler, the proposal draws $\lambda'$ without reference to the current state:

$$q(\lambda' \mid \lambda) = q(\lambda'), \qquad q(\lambda \mid \lambda') = q(\lambda)$$

The general MH ratio therefore simplifies to:

$$A(\lambda, \lambda') = \min\!\left(1,\; \frac{p(\lambda' \mid x)\, q(\lambda)}{p(\lambda \mid x)\, q(\lambda')}\right) = \min\!\left(1,\; \frac{w(\lambda')}{w(\lambda)}\right)$$

where $w(\lambda) = p(\lambda \mid x) / q(\lambda)$ is the unnormalised importance weight. The sampler accepts a proposal whenever the proposed state has a higher weight than the current state, and accepts with probability $w(\lambda')/w(\lambda)$ otherwise.

### (b) Proposal Choice and Diagnostics

The proposal is $q(\lambda') = \text{Gamma}(20, 1/7)$ (shape 20, scale $1/7$), which has mean $20 \times 1/7 = 20/7$ — exactly the posterior mean. Since the target is $\text{Gamma}(20, 7)$ (shape 20, rate 7, equivalently scale $1/7$), the proposal and posterior are in fact the same distribution. Consequently, the importance weight $w(\lambda) = p(\lambda \mid x)/q(\lambda)$ is constant in $\lambda$, and the acceptance ratio is identically 1.

The diagnostic comparison between the two samplers is as follows.

**Acceptance rate.** The RWMH achieved approximately 32% (within the target window). The independence sampler accepted every proposal (100%), reflecting the exact proposal-posterior match.

**Autocorrelation.** The RWMH chains showed gradual decay, with autocorrelation dropping to noise after roughly 10 lags. The independence sampler's autocorrelation fell to zero at lag 1, as successive draws are independent by construction.

**Effective sample size.** With 2000 post-burn-in draws per chain across 4 chains, the total ESS for the RWMH was approximately 210, compared to approximately 330 for the independence sampler.

**Overall verdict.** The independence sampler performs vastly better here. The reason is structural: because $q$ is identical to the posterior, every draw is an independent posterior sample and no information is lost to correlation or rejection. In practice, however, a proposal this well-matched is rarely available — it requires the posterior to be a standard distribution in closed form. The independence sampler's advantage is therefore problem-specific, and the RWMH remains the more general tool.
