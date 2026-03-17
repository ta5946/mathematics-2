### 1. Irreducibility
A Markov chain is irreducible if it is possible to reach any state from any other state, meaning all states communicate and form a single class. 

By observing the transition matrix $K$, we can trace the strictly positive transition probabilities:
* From $s_1$, we can transition to $s_2$ since $K_{12} = 0.6 > 0$.
* From $s_2$, we can transition to $s_4$ since $K_{24} = 0.1 > 0$.
* From $s_4$, we can transition back to $s_1$ since $K_{41} = 0.4 > 0$.

This establishes a cycle: $s_1 \to s_2 \to s_4 \to s_1$. Furthermore, $s_3$ is reachable from $s_1$ ($K_{13} = 0.3$) and can reach $s_2$ ($K_{32} = 0.5$), integrating it fully into the communicating class. Since all states communicate, the chain is irreducible.

---

### 2. Stationary Distribution
The unique stationary distribution is the normalized row vector $\pi$ that satisfies the equation:
$$\pi K = \pi$$
Subject to the constraint that the probabilities sum to 1:
$$\sum_{j=1}^{4} \pi_j = 1$$
Using software to solve this linear system (finding the left eigenvector associated with the eigenvalue 1), we obtain the following stationary probabilities:
$$\pi \approx [0.1425, 0.3726, 0.3507, 0.1342]$$

---

### 3. Expected Return Time
For an irreducible Markov chain, the Ergodic Theorem states that the long-run proportion of time the chain spends in state $i$ converges to its stationary probability $\pi_i$:
$$\lim_{n\to\infty} \frac{1}{n} \sum_{t=1}^n \mathbb{1}_{\{X_t = i\}} = \pi_i$$

Let $N_i(n)$ be the total number of visits to state $i$ up to time $n$. The proportion of time spent in state $i$ can be written as:
$$\lim_{n\to\infty} \frac{N_i(n)}{n} = \pi_i$$

The total elapsed time $n$ is approximately the sum of the $N_i(n)$ individual return times, denoted as $(T_i)_k$. The empirical average of these return times is the total time divided by the number of visits:
$$\text{Average Return Time} = \frac{1}{N_i(n)} \sum_{k=1}^{N_i(n)} (T_i)_k = \frac{n}{N_i(n)}$$

Taking the limit as the number of steps $n \to \infty$, the average return time converges to the expected return time $E[T_i]$:
$$E[T_i] = \lim_{n\to\infty} \frac{n}{N_i(n)} = \frac{1}{\pi_i}$$

Using our calculated stationary distribution, the expected return times $E[T_i]$ for states $s_1$ through $s_4$ are approximately **7.02**, **2.68**, **2.85**, and **7.45** steps, respectively.

---

### 4. Limiting Value (Ergodic Theorem)
By the Ergodic Theorem for irreducible, positive recurrent Markov chains, the long-run time average of a function $g$ applied to the sequence of states converges to the expected value of $g$ under the stationary distribution $\pi$:
$$\lim_{n\to\infty} \frac{1}{n} \sum_{t=1}^{n} g(X_t) = \sum_{j=1}^{4} \pi_j g(s_j)$$

Given the function values $g(1)=1$, $g(2)=0$, $g(3)=2$, $g(4)=1$ and our calculated distribution $\pi \approx [0.1425, 0.3726, 0.3507, 0.1342]$, we substitute these into the sum:
$$\sum_{j=1}^{4} \pi_j g(s_j) \approx (0.1425 \cdot 1) + (0.3726 \cdot 0) + (0.3507 \cdot 2) + (0.1342 \cdot 1)$$
$$\sum_{j=1}^{4} \pi_j g(s_j) \approx 0.1425 + 0 + 0.7014 + 0.1342 = 0.9781$$

---

### 5. Aperiodicity
A state $s_i$ is aperiodic if the greatest common divisor (GCD) of the set of all its possible return times is 1. A sufficient condition for a state to be aperiodic is the presence of a self-loop, meaning $K_{ii} > 0$, as this guarantees a possible return time of 1 step.

Looking at the main diagonal of the transition matrix $K$:
* $K_{11} = 0.1 > 0$
* $K_{22} = 0.3 > 0$
* $K_{33} = 0.3 > 0$
* $K_{44} = 0.2 > 0$

Every state has a strictly positive probability of remaining in itself for the next step, ensuring that every state has a period of 1. Because aperiodicity is a class property in an irreducible chain, and all states are aperiodic, the entire Markov chain is aperiodic.
