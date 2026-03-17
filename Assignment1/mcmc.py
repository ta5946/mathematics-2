import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("=== 1 Importance Sampling ===\n")

# Integrand function for target integral
def f(x):
    return x**(1/3) * np.exp(-x**2)

# Normalization constant for proposal density
c = 4 / (3 * 2**(4/3))

# Calculate actual integral with scipy
true_value, _ = quad(f, 0, 2)
print("--- Part 1: Actual Integral ---")
print(f"Mean: {true_value:.8f}")
print(f"Std Dev: 0.00000000\n")

# Plot integrand
x_plot = np.linspace(0, 2, 500)
y_plot = f(x_plot)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label='$f(x) = x^{1/3}e^{-x^2}$')
plt.title('Integrand Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Global simulation parameters
n = 10**6
repeats = 10

# Monte Carlo estimator
mc_estimates = []

for _ in range(repeats):
    # Standard uniform sampling
    x_mc = np.random.uniform(0, 2, n)
    # Mean of heights scaled by interval width
    estimate = 2 * np.mean(f(x_mc))
    mc_estimates.append(estimate)

print("--- Part 2: Standard Monte Carlo ---")
print(f"Mean: {np.mean(mc_estimates):.8f}")
print(f"Std Dev: {np.std(mc_estimates, ddof=1):.8f}\n")

# Importance Sampling estimator
is_estimates = []

for _ in range(repeats):
    # Probability percentiles from uniform distribution
    u = np.random.uniform(0, 1, n)
    # Samples generated via inverse transform sampling
    x_is = 2 * u**(3/4)
    # Ratio of target function to proposal density
    is_val = np.exp(-x_is**2) / c
    is_estimates.append(np.mean(is_val))

print("--- Part 3: Importance Sampling ---")
print(f"Mean: {np.mean(is_estimates):.8f}")
print(f"Std Dev: {np.std(is_estimates, ddof=1):.8f}\n")


print("=== 2 Markov Chains ===\n")

# Define transition matrix
K = np.array([
    [0.1, 0.6, 0.3, 0.0],
    [0.2, 0.3, 0.4, 0.1],
    [0.0, 0.5, 0.3, 0.2],
    [0.4, 0.0, 0.4, 0.2]
])

# Solve for stationary distribution
# Evaluate eigenvector identity for transposed matrix
eigenvalues, eigenvectors = np.linalg.eig(K.T)

# Find eigenvector for unit eigenvalue
index = np.argmin(np.abs(eigenvalues - 1.0))
pi = eigenvectors[:, index].real

# Normalize probabilities to sum to 1
pi = pi / np.sum(pi)

print("--- Part 2: Stationary Distribution ---")
print(f"pi: {pi}")

# Compute expected return times
expected_return = 1 / pi

print("\n--- Part 3: Expected Return Times ---")
print(f"E[T_i]: {expected_return}")

# Calculate limit of time average of function g
g = np.array([1, 0, 2, 1])
limiting_value = np.dot(pi, g)

print("\n--- Part 4: Limiting Value ---")
print(f"Limit: {limiting_value:.4f}")
