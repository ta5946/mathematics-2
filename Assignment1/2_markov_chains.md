# 2. Markov Chains

The chain $(X_n)_{n \in \mathbb{N}}$ evolves on the state space $S = \{s_1, s_2, s_3, s_4\}$ with transition matrix:

$$K = \begin{pmatrix} 0.1 & 0.6 & 0.3 & 0.0 \\ 0.2 & 0.3 & 0.4 & 0.1 \\ 0.0 & 0.5 & 0.3 & 0.2 \\ 0.4 & 0.0 & 0.4 & 0.2 \end{pmatrix}$$

---

## 2.1 Irreducibility

A chain is **irreducible** if every state communicates with every other state, i.e. for all $i, j \in S$ there exists $n \geq 1$ such that $K^n_{ij} > 0$.

The following strictly positive entries of $K$ establish a cycle through all four states:

$$s_1 \xrightarrow{K_{12}=0.6} s_2 \xrightarrow{K_{24}=0.1} s_4 \xrightarrow{K_{41}=0.4} s_1$$

State $s_3$ is reached from $s_1$ via $K_{13} = 0.3 > 0$, and returns to the cycle via $K_{32} = 0.5 > 0$. Since every state can reach and be reached from every other state, all four states form a single communicating class. The chain is therefore irreducible.

---

## 2.2 Stationary Distribution

The stationary distribution is the unique probability vector $\pi$ satisfying:

$$\pi K = \pi, \qquad \sum_{j=1}^{4} \pi_j = 1$$

This is equivalent to finding the left eigenvector of $K$ associated with eigenvalue 1, normalised to sum to 1. Solving numerically:

$$\pi \approx [0.1425,\; 0.3726,\; 0.3507,\; 0.1342]$$

---

## 2.3 Expected Return Times

**Claim:** For each state $i$, the expected return time satisfies $E[T_i] = 1/\pi_i$.

**Proof.** Since the chain is irreducible and the state space is finite, it is positive recurrent, and the ergodic theorem applies. Let $N_i(n)$ denote the number of visits to state $i$ in the first $n$ steps. By the ergodic theorem:

$$\frac{N_i(n)}{n} \xrightarrow{n \to \infty} \pi_i \quad \text{almost surely}$$

Between consecutive visits to $i$, the elapsed time is exactly one return time $T_i^{(k)}$. By the strong law of large numbers applied to the renewal process, the empirical mean of these return times converges almost surely to $E[T_i]$:

$$\frac{1}{N_i(n)} \sum_{k=1}^{N_i(n)} T_i^{(k)} \xrightarrow{n \to \infty} E[T_i]$$

The left-hand side also equals $n / N_i(n)$, since the $N_i(n)$ return intervals partition the $n$ elapsed steps. Taking the limit:

$$E[T_i] = \lim_{n \to \infty} \frac{n}{N_i(n)} = \frac{1}{\pi_i} \qquad \square$$

Substituting the stationary probabilities gives the expected return times:

| State | $\pi_i$ | $E[T_i] = 1/\pi_i$ |
|-------|---------|---------------------|
| $s_1$ | 0.1425 | 7.02 steps |
| $s_2$ | 0.3726 | 2.68 steps |
| $s_3$ | 0.3507 | 2.85 steps |
| $s_4$ | 0.1342 | 7.45 steps |

---

## 2.4 Limiting Time Average

For an irreducible, positive recurrent Markov chain, the ergodic theorem gives:

$$\lim_{n\to\infty} \frac{1}{n} \sum_{t=1}^{n} g(X_t) = \mathbb{E}_\pi[g] = \sum_{j=1}^{4} \pi_j\, g(s_j) \quad \text{almost surely}$$

for any bounded function $g: S \to \mathbb{R}$. With $g(s_1)=1$, $g(s_2)=0$, $g(s_3)=2$, $g(s_4)=1$ and $\pi \approx [0.1425, 0.3726, 0.3507, 0.1342]$:

$$\mathbb{E}_\pi[g] = (0.1425)(1) + (0.3726)(0) + (0.3507)(2) + (0.1342)(1)$$
$$= 0.1425 + 0 + 0.7014 + 0.1342 = \boxed{0.9781}$$

---

## 2.5 Aperiodicity

The **period** of state $i$ is $d(i) = \gcd\{n \geq 1 : K^n_{ii} > 0\}$. A state is aperiodic if $d(i) = 1$.

A sufficient condition for aperiodicity is $K_{ii} > 0$: a self-loop guarantees that return in exactly 1 step is possible, forcing $\gcd$ to divide 1. Inspecting the diagonal of $K$:

$$K_{11} = 0.1 > 0, \quad K_{22} = 0.3 > 0, \quad K_{33} = 0.3 > 0, \quad K_{44} = 0.2 > 0$$

Every state has a self-loop, so every state is aperiodic. Since the chain is irreducible, aperiodicity is a class property: if one state is aperiodic, all states are. The chain is therefore aperiodic.

Together with irreducibility and positive recurrence, aperiodicity implies that $K^n_{ij} \to \pi_j$ as $n \to \infty$ for all $i, j$, i.e. the chain converges to $\pi$ from any starting state.
