# Mathematics 2: Homework GD.1

## Theory part

---

## Problem 1: Prove equivalence 1. ⟺ 4. of Proposition 2.3

We want to prove that the following are equivalent:

**1.** $f$ is convex, i.e. $\forall x, y \in D, \forall t \in [0,1]$:
$$f(tx + (1-t)y) \leq tf(x) + (1-t)f(y)$$

**4.** (Jensen's inequality) For all $x_1, \ldots, x_k \in D$ and $\alpha_1, \ldots, \alpha_k \in [0,1]$ with $\sum_{i=1}^{k} \alpha_i = 1$:
$$f\left(\sum_{i=1}^{k} \alpha_i x_i\right) \leq \sum_{i=1}^{k} \alpha_i f(x_i)$$

### Direction 4. ⟹ 1. (trivial)

Take $k=2$, $\alpha_1 = t$, $\alpha_2 = 1-t$, $x_1 = x$, $x_2 = y$. Then $\sum \alpha_i = 1$ and statement 4 becomes exactly statement 1. $\square$

### Direction 1. ⟹ 4. (by induction on k)

**Base case (k=2):** Statement 4 reduces to $f(\alpha_1 x_1 + \alpha_2 x_2) \leq \alpha_1 f(x_1) + \alpha_2 f(x_2)$ with $\alpha_1 + \alpha_2 = 1$. Setting $t = \alpha_1$ gives exactly statement 1. $\checkmark$

**Edge case:** If $\alpha_{k+1} = 1$, then all other $\alpha_i = 0$, so the inequality reduces to $f(x_{k+1}) \leq f(x_{k+1})$, which is trivially true. So we assume $\alpha_{k+1} < 1$ from now on.

**Inductive step:** Assume statement 4 holds for $k$. We prove it for $k+1$.

Define
$$S := \sum_{i=1}^{k} \alpha_i = 1 - \alpha_{k+1} > 0, \qquad z := \sum_{i=1}^{k} \frac{\alpha_i}{S} x_i$$

and split the sum:
$$\sum_{i=1}^{k+1} \alpha_i x_i = S \cdot z + \alpha_{k+1} \cdot x_{k+1}$$

The rescaled weights $\alpha_i/S$ are valid: each lies in $[0,1]$ since $0 \leq \alpha_i \leq S$, and $\sum_{i=1}^{k} \frac{\alpha_i}{S} = \frac{S}{S} = 1$. The outer combination is valid since $S + \alpha_{k+1} = 1$.

The role of $S$ is to make **both** combinations valid convex combinations at the same time: the outer one satisfies $S + \alpha_{k+1} = 1$ so we can use the base case, and the inner one satisfies $\sum \alpha_i / S = 1$ so we can use the inductive hypothesis.

Using the base case on the outer 2-point combination:
$$f\left(\sum_{i=1}^{k+1} \alpha_i x_i\right) = f(S \cdot z + \alpha_{k+1} \cdot x_{k+1}) \leq S \cdot f(z) + \alpha_{k+1} \cdot f(x_{k+1})$$

Using the inductive hypothesis on $f(z)$:
$$f(z) \leq \sum_{i=1}^{k} \frac{\alpha_i}{S} f(x_i)$$

Combining:
$$f\left(\sum_{i=1}^{k+1} \alpha_i x_i\right) \leq S \cdot \sum_{i=1}^{k} \frac{\alpha_i}{S} f(x_i) + \alpha_{k+1} f(x_{k+1}) = \sum_{i=1}^{k+1} \alpha_i f(x_i)$$

The $S$ cancels cleanly, completing the induction. $\square$

---

## Problem 2: Constants L, α, β for f(x,y) = x² + eˣ + y² − xy on K = [−2,2] × [−2,2]

### Gradient and Hessian

$$\nabla f = \begin{pmatrix} 2x + e^x - y \\ 2y - x \end{pmatrix}, \qquad H_f = \nabla^2 f = \begin{pmatrix} 2 + e^x & -1 \\ -1 & 2 \end{pmatrix}$$

Note that the Hessian depends only on $x$, not on $y$.

### Eigenvalues of the Hessian

Setting $\det(H_f - \lambda I) = 0$:
$$(2 + e^x - \lambda)(2 - \lambda) - 1 = 0$$
$$\lambda^2 - (4 + e^x)\lambda + (3 + 2e^x) = 0$$

Using the quadratic formula:
$$\lambda = \frac{(4 + e^x) \pm \sqrt{(4 + e^x)^2 - 4(3 + 2e^x)}}{2} = \frac{(4 + e^x) \pm \sqrt{e^{2x} + 4}}{2}$$

The square root term $\sqrt{e^{2x}+4}$ is always positive, so the $+$ sign gives the larger root and the $-$ sign gives the smaller:
$$\lambda_{max}(x) = \frac{(4 + e^x) + \sqrt{e^{2x} + 4}}{2}, \quad \lambda_{min}(x) = \frac{(4 + e^x) - \sqrt{e^{2x} + 4}}{2}$$

### Finding β (optimal smoothness constant)

By **Proposition 2.8**, $f$ is $\beta$-smooth $\Longleftrightarrow \lambda_i(H_f(x)) \leq \beta$ for $\forall i, \forall x \in K$. This means $\beta$ must satisfy:
$$\beta \geq \lambda_{max}(x) \quad \forall x \in K$$

The tightest (smallest) valid $\beta$ is therefore:
$$\beta_{opt} = \max_{x \in K} \lambda_{max}(x)$$

Both terms $(4+e^x)$ and $\sqrt{e^{2x}+4}$ are increasing in $x$, so $\lambda_{max}$ is maximized at $x = 2$:
$$\boxed{\beta_{opt} = \frac{(4 + e^2) + \sqrt{e^4 + 4}}{2} \approx 9.522}$$

### Finding α (optimal strong convexity constant)

By **Proposition 2.11**, $f$ is $\alpha$-strongly convex $\Longleftrightarrow \lambda_i(H_f(x)) \geq \alpha$ for $\forall i, \forall x \in K$. This means $\alpha$ must satisfy:
$$\alpha \leq \lambda_{min}(x) \quad \forall x \in K$$

The tightest (largest) valid $\alpha$ is therefore:
$$\alpha_{opt} = \min_{x \in K} \lambda_{min}(x)$$

Checking endpoints:
- At $x = -2$: $\lambda_{min} = \frac{(4 + e^{-2}) - \sqrt{e^{-4} + 4}}{2} \approx 1.065$
- At $x = 2$: $\lambda_{min} = \frac{(4 + e^2) - \sqrt{e^4 + 4}}{2} \approx 1.867$

So $\lambda_{min}$ is smallest at $x = -2$:
$$\boxed{\alpha_{opt} = \frac{(4 + e^{-2}) - \sqrt{e^{-4} + 4}}{2} \approx 1.065}$$

### Convexity

By the last part of **Proposition 2.3**, $f$ is convex $\Longleftrightarrow$ all eigenvalues of $H_f$ are non-negative everywhere on $K$. Since $\alpha_{opt} \approx 1.065 > 0$, all eigenvalues of $H_f$ are strictly positive on $K$, so $H_f$ is positive definite everywhere on $K$, hence $f$ is $\alpha_{opt}$-strongly convex on $K$, and in particular strictly convex. $\square$

### Finding L (optimal Lipschitz constant)

By **Proposition 2.6**, $f$ is $L$-Lipschitz $\Longleftrightarrow \|\nabla f(x,y)\| \leq L$ for $\forall (x,y) \in K$. This means $L$ must satisfy:
$$L \geq \|\nabla f(x,y)\| \quad \forall (x,y) \in K$$

The tightest (smallest) valid $L$ is therefore:
$$L_{opt} = \max_{(x,y) \in K} \|\nabla f(x,y)\|$$

Equivalently we maximize:
$$g(x,y) = \|\nabla f\|^2 = (2x + e^x - y)^2 + (2y - x)^2$$

$g$ is smooth on the compact set $K$, so its maximum is attained. The interior stationary points of $g$ satisfy $\nabla g = 0$, which (expanding) forces $\nabla f = 0$; but since $f$ is strongly convex on $K$ (previous part), the only such point is the global minimizer of $f$, where $\|\nabla f\| = 0$, which is clearly not the max. Hence the max lies on the boundary of $K$. On each edge of the square, $g$ reduces to a smooth function of one variable whose derivative is a sum of an exponential and a polynomial; a quick check shows no interior-edge maximum beats the corners, so the max is attained at a vertex.

The first component is maximized at $(x,y) = (2, -2)$ with value $6 + e^2$, and the second at $(-2, 2)$ with value $6$. These occur at **different corners**, so we check all four corners of $K$.

Evaluating $\|\nabla f\|$ at each corner:

| Corner | $\|\nabla f\|$ |
|--------|---------------|
| $(2, -2)$ | $\sqrt{(6+e^2)^2 + 36} \approx 14.67$ |
| $(-2, 2)$ | $\sqrt{(e^{-2}-6)^2 + 36} \approx 8.34$ |
| $(2, 2)$ | $\sqrt{(2+e^2)^2 + 4} \approx 9.60$ |
| $(-2, -2)$ | $\sqrt{(e^{-2}-2)^2 + 4} \approx 2.70$ |

The maximum is attained at corner $(2, -2)$:
$$\boxed{L_{opt} = \sqrt{(6 + e^2)^2 + 36} \approx 14.67}$$

---

## Problem 3: Projection formulas onto convex sets

Recall from the **PGD** definition in the notes that the projection $\pi_K : \mathbb{R}^2 \to K$ onto a closed convex set $K$ maps each $p$ to its closest point in $K$. By **Lemma 3.2** this closest point is unique. If $p \in K$ already, then $\pi_K(p) = p$; otherwise the projection lies on the boundary.

We write $\pi_{[a,b]}$ for the projection onto an interval $[a, b]$:
$$\pi_{[a,b]}(t) = \begin{cases} a & \text{if } t < a \\ t & \text{if } t \in [a,b] \\ b & \text{if } t > b \end{cases}$$

### (a) Disk: $K_1 = \{(x,y) : x^2 + y^2 \leq 1.5\}$

Let $r = \sqrt{1.5}$. By symmetry, if $p$ is outside the disk, the closest point lies on the line from the origin to $p$, at radius $r$:

$$\boxed{\pi_{K_1}(p) = \begin{cases} p & \text{if } \|p\| \leq r \\ r \cdot \dfrac{p}{\|p\|} & \text{if } \|p\| > r \end{cases}}$$

### (b) Square: $K_2 = [-1,1] \times [-1,1]$

Since $K_2$ is a Cartesian product of intervals, the projection factors coordinate-wise:

$$\boxed{\pi_{K_2}(p_1, p_2) = \big(\pi_{[-1,1]}(p_1),\ \pi_{[-1,1]}(p_2)\big)}$$

### (c) Triangle: $K_3$ with vertices $A=(-1,-1)$, $B=(1.5,-1)$, $C=(-1,1.5)$

```
     C (-1, 1.5)
     |\
     | \
     |  \
     |   \
     |    \
     |     \
     |______\
  A(-1,-1)  B(1.5,-1)
```

The triangle is described by three constraints:
$$x \geq -1, \quad y \geq -1, \quad x + y \leq 0.5$$

The hypotenuse $BC$ lies on the line $x + y = 0.5$ (slope $-1$, verified by plugging in $B$ and $C$).

**Derivation of the hypotenuse projection:** This case applies when $p$ lies on the far side of line $BC$ from the triangle, i.e. $p_1 + p_2 > 0.5$. The projection of $p = (p_1, p_2)$ onto the line $x+y = 0.5$ is
$$p - \frac{p_1+p_2-0.5}{2}\begin{pmatrix}1\\1\end{pmatrix} = \left(\frac{p_1-p_2+0.5}{2},\ \frac{p_2-p_1+0.5}{2}\right)$$

This point lies on segment $BC$ $\Longleftrightarrow$ its first coordinate is in $[-1, 1.5]$, which is equivalent to $|p_1 - p_2| \leq 2.5$. If $p_1 - p_2 > 2.5$ we snap to $B$; if $p_2 - p_1 > 2.5$ we snap to $C$.

**Projection formula** (by case analysis on which constraints are violated):

$$\boxed{\pi_{K_3}(p) = \begin{cases}
p & \text{if } p \in K_3 \\[6pt]
(-1, -1) & \text{if } p_1 < -1,\ p_2 < -1 \\[6pt]
(-1,\ \pi_{[-1, 1.5]}(p_2)) & \text{if } p_1 < -1,\ p_2 \geq -1 \\[6pt]
(\pi_{[-1, 1.5]}(p_1),\ -1) & \text{if } p_2 < -1,\ p_1 \geq -1 \\[6pt]
\left(\dfrac{p_1-p_2+0.5}{2},\ \dfrac{p_2-p_1+0.5}{2}\right) & \text{if } p_1 \geq -1,\ p_2 \geq -1,\ p_1+p_2 > 0.5,\ |p_1-p_2| \leq 2.5 \\[6pt]
(1.5, -1) & \text{if } p_1+p_2 > 0.5,\ p_1 - p_2 > 2.5 \\[6pt]
(-1, 1.5) & \text{if } p_1+p_2 > 0.5,\ p_2 - p_1 > 2.5
\end{cases}}$$

---

## Problem 4: One step of gradient descent on f(x,y) = x² + 2y²

First compute the gradient:
$$\nabla f(x, y) = \begin{pmatrix} 2x \\ 4y \end{pmatrix}$$

At $x_1 = (1, 1)$:
$$\nabla f(x_1) = \begin{pmatrix} 2 \\ 4 \end{pmatrix}, \qquad f(x_1) = 1^2 + 2 \cdot 1^2 = 3$$

The GD step is:
$$x_2 = x_1 - \gamma \nabla f(x_1) = \begin{pmatrix} 1 \\ 1 \end{pmatrix} - \gamma \begin{pmatrix} 2 \\ 4 \end{pmatrix} = \begin{pmatrix} 1 - 2\gamma \\ 1 - 4\gamma \end{pmatrix}$$

### (a) Minimize f(x₂)

Plug $x_2$ into $f$:
$$f(x_2) = (1-2\gamma)^2 + 2(1-4\gamma)^2$$

Expand each term:
$$(1-2\gamma)^2 = 1 - 4\gamma + 4\gamma^2$$
$$2(1-4\gamma)^2 = 2(1 - 8\gamma + 16\gamma^2) = 2 - 16\gamma + 32\gamma^2$$

Sum:
$$f(x_2) = 3 - 20\gamma + 36\gamma^2$$

This is a quadratic in $\gamma$, minimized where its derivative is zero:
$$\frac{d}{d\gamma} f(x_2) = -20 + 72\gamma = 0 \implies \gamma = \frac{20}{72} = \frac{5}{18}$$

Plugging back:
$$x_2 = \begin{pmatrix} 1 - \frac{10}{18} \\ 1 - \frac{20}{18} \end{pmatrix} = \begin{pmatrix} \frac{4}{9} \\ -\frac{1}{9} \end{pmatrix}$$

$$f(x_2) = 3 - 20 \cdot \frac{5}{18} + 36 \cdot \frac{25}{324} = 3 - \frac{50}{9} + \frac{25}{9} = 3 - \frac{25}{9} = \boxed{\frac{2}{9} \approx 0.222}$$

So one step of GD reduces the function value from $3$ to $\frac{2}{9}$.

### (b) Minimize distance to x*

The global minimum of $f$ is at $x^* = (0, 0)$ (both $x^2$ and $2y^2$ are non-negative and vanish only at the origin).

The squared distance from $x_2$ to $x^*$ is:
$$h(\gamma) := \|x_2 - x^*\|^2 = (1-2\gamma)^2 + (1-4\gamma)^2$$

Expand:
$$(1-2\gamma)^2 = 1 - 4\gamma + 4\gamma^2$$
$$(1-4\gamma)^2 = 1 - 8\gamma + 16\gamma^2$$

Sum:
$$h(\gamma) = 2 - 12\gamma + 20\gamma^2$$

Minimize:
$$h'(\gamma) = -12 + 40\gamma = 0 \implies \gamma = \frac{12}{40} = \frac{3}{10}$$

Plugging back:
$$x_2 = \begin{pmatrix} 1 - \frac{6}{10} \\ 1 - \frac{12}{10} \end{pmatrix} = \begin{pmatrix} \frac{2}{5} \\ -\frac{1}{5} \end{pmatrix}$$

$$\|x_2 - x^*\|^2 = 2 - 12 \cdot \frac{3}{10} + 20 \cdot \frac{9}{100} = 2 - \frac{18}{5} + \frac{9}{5} = 2 - \frac{9}{5} = \frac{1}{5}$$

$$\boxed{\|x_2 - x^*\| = \frac{1}{\sqrt{5}} \approx 0.447}$$

### Observation

The two optimal learning rates differ ($\frac{5}{18} \approx 0.278$ vs $\frac{3}{10} = 0.3$), because minimizing the function value and minimizing the distance to $x^*$ are **not the same objective** when the level sets of $f$ are non-circular. Here the level sets of $f(x,y) = x^2 + 2y^2$ are ellipses stretched along the $x$-axis, so moving closer to $x^*$ in Euclidean distance is not the same as moving to a lower function value.

---

## Programming part

---

## Problem 5: PGD on f(x,y) = x² + eˣ + y² − xy

We run PGD from $x_1 = (-1, 1)$ for $k = 10$ steps on each of the three domains from Problem 3, using the three learning rates from **Theorem 3.3** and constants $\alpha, \beta, L, \kappa$ from Problem 2. Let $R = \|x_1 - x^*\|$.

The three cases give:

**Case 1** ($L$-Lipschitz): $\gamma_1 = \frac{R}{L\sqrt{k}} \approx 0.0289$, bound on $f\!\left(\frac{1}{k}\sum_{i=1}^k x_i\right) - f(x^*)$:
$$B_1 = \frac{LR}{\sqrt{k}} \approx 6.227$$

**Case 2** ($\beta$-smooth): $\gamma_2 = \frac{1}{\beta} \approx 0.1050$, bound on $f(x_k) - f(x^*)$:
$$B_2 = \frac{3\beta R^2 + f(x_1) - f(x^*)}{k} \approx 5.404$$

**Case 3** ($\alpha$-strongly convex and $\beta$-smooth): $\gamma_3 = \frac{1}{\beta} \approx 0.1050$, bound on $f(x_{k+1}) - f(x^*)$:
$$B_3 = \frac{\beta}{2}\left(\frac{\kappa-1}{\kappa}\right)^{2k} R^2 \approx 0.799$$

Note that $\gamma_2 = \gamma_3 = \frac{1}{\beta}$: both cases use the $\beta$-smooth rate. The bounds differ because Case 3 additionally exploits $\alpha$-strong convexity through $\kappa = \beta/\alpha$, giving a much tighter guarantee.

The rows of the table below are identical across all three domains; this is not a copy-paste error. The reason is explained in the discussion that follows the table.

| Domain | Rate | $\gamma$ | $f(x) - f(x^*)$ | Bound |
|--------|------|-------:|----------------:|------:|
| Disk | (1) $L$-Lipschitz | 0.0289 | 0.887934 | 6.2271 |
| Disk | (2) $\beta$-smooth | 0.1050 | 0.015181 | 5.4035 |
| Disk | (3) $\alpha$-SC + $\beta$-smooth | 0.1050 | 0.015181 | 0.7992 |
| Square | (1) $L$-Lipschitz | 0.0289 | 1.004952 | 6.2271 |
| Square | (2) $\beta$-smooth | 0.1050 | 0.015181 | 5.4035 |
| Square | (3) $\alpha$-SC + $\beta$-smooth | 0.1050 | 0.015181 | 0.7992 |
| Triangle | (1) $L$-Lipschitz | 0.0289 | 1.004952 | 6.2271 |
| Triangle | (2) $\beta$-smooth | 0.1050 | 0.015181 | 5.4035 |
| Triangle | (3) $\alpha$-SC + $\beta$-smooth | 0.1050 | 0.015181 | 0.7992 |

$$\boxed{f(x) - f(x^*) < B_i \text{ in every row, as guaranteed by Theorem 3.3}}$$

Results are identical across all three domains because $x^* \approx (-0.433, -0.216)$ lies in the interior of all three sets, so the projection never fires. Case 1 performs worst ($f(x) - f(x^*) \approx 0.89$) due to its very small step size. Cases 2 and 3 achieve the same difference ($\approx 0.015$) since $\gamma_2 = \gamma_3$. **Case 3 has the best theoretical guarantee**: $B_3 \approx 0.799$ is the tightest bound, because our function satisfies strong convexity and we can exploit this via $\kappa$.

---

## Problem 6: Minimizing Hartmann-3 on $[0,1]^3$

We minimize
$$f(z) = -\sum_{i=1}^{4} c_i \exp\left(-\sum_{j=1}^{3} a_{i,j}(z_j - p_{i,j})^2\right)$$
using PGD with projection onto $[0,1]^3$. The gradient is:
$$\frac{\partial f}{\partial z_k} = \sum_{i=1}^{4} 2\, c_i\, a_{i,k}\,(z_k - p_{i,k})\, \exp\left(-\sum_{j=1}^{3} a_{i,j}(z_j - p_{i,j})^2\right)$$

The projection onto $[0,1]^3$ factors coordinate-wise, identical to the square projection from Problem 3 with $a=0$, $b=1$:
$$\pi_{[0,1]^3}(z) = \big(\pi_{[0,1]}(z_1),\ \pi_{[0,1]}(z_2),\ \pi_{[0,1]}(z_3)\big)$$

Starting from $z_1 = (0.5, 0.5, 0.5)$ with a standard learning rate $\gamma = 0.01$, PGD converges in **2203 steps** to:
$$z^* \approx (0.1146,\ 0.5556,\ 0.8525)$$
$$\boxed{f(z^*) - f_{\text{target}} \approx 4.44 \times 10^{-15}}$$

This is floating point precision. The function is not convex so the theoretical guarantees from the course do not apply, but starting from the center of the box is sufficient to reach the global minimum.
