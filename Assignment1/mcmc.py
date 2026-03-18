import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import arviz as az

# Set global random seed for complete reproducibility
np.random.seed(42)

print("=== 1 Importance Sampling ===")


# Target integrand function
def f(x):
    return x ** (1 / 3) * np.exp(-x ** 2)


# Normalization constant for proposal distribution
c = 4 / (3 * 2 ** (4 / 3))

# Calculate exact integral using numerical quadrature
true_value, _ = quad(f, 0, 2)

print("\n--- Part 1: Actual Integral ---")
print(f"Mean: {true_value:.8f}")
print(f"Std Dev: 0.00000000")

# Setup values for integrand visualization
x_plot = np.linspace(0, 2, 500)
y_plot = f(x_plot)

# Render plot of the target function
plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label='$f(x) = x^{1/3}e^{-x^2}$')
plt.title('Integrand Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Base parameters for Monte Carlo simulations
n = 10 ** 6
repeats = 10

# Calculate standard Monte Carlo estimates
mc_estimates = []
for _ in range(repeats):
    # Draw from uniform distribution
    x_mc = np.random.uniform(0, 2, n)
    # Scale mean of evaluations by interval width
    estimate = 2 * np.mean(f(x_mc))
    mc_estimates.append(estimate)

print("\n--- Part 2: Standard Monte Carlo ---")
print(f"Mean: {np.mean(mc_estimates):.8f}")
print(f"Std Dev: {np.std(mc_estimates, ddof=1):.8f}")

# Calculate importance sampling estimates
is_estimates = []
for _ in range(repeats):
    # Draw base uniform probabilities
    u = np.random.uniform(0, 1, n)
    # Apply inverse transform sampling
    x_is = 2 * u ** (3 / 4)
    # Calculate importance weights
    is_val = np.exp(-x_is ** 2) / c
    is_estimates.append(np.mean(is_val))

print("\n--- Part 3: Importance Sampling ---")
print(f"Mean: {np.mean(is_estimates):.8f}")
print(f"Std Dev: {np.std(is_estimates, ddof=1):.8f}")

print("\n=== 2 Markov Chains ===")

# Define transition probability matrix

K = np.array([
    [0.1, 0.6, 0.3, 0.0],
    [0.2, 0.3, 0.4, 0.1],
    [0.0, 0.5, 0.3, 0.2],
    [0.4, 0.0, 0.4, 0.2]
])

# Extract eigenvalues and eigenvectors for the transposed matrix
eigenvalues, eigenvectors = np.linalg.eig(K.T)

# Isolate the eigenvector corresponding to the stationary state
index = np.argmin(np.abs(eigenvalues - 1.0))
pi = eigenvectors[:, index].real

# Normalize the stationary distribution vector
pi = pi / np.sum(pi)

print("\n--- Part 2: Stationary Distribution ---")
print(f"pi: {pi}")

# Compute expected return times for each state
expected_return = 1 / pi

print("\n--- Part 3: Expected Return Times ---")
print(f"E[T_i]: {expected_return}")

# Calculate limit of time average for specific function
g = np.array([1, 0, 2, 1])
limiting_value = np.dot(pi, g)

print("\n--- Part 4: Limiting Value ---")
print(f"Limit: {limiting_value:.4f}")

print("\n=== 3 Metropolis-Hastings Algorithm ===")

# Define prior parameters and sample statistics
sum_x, n_data = 19, 6
alpha, beta = 1, 1


# Metropolis-Hastings transition kernel
def run_mh(sigma_sq, iterations, start_val):
    lam = start_val
    accepts = 0
    chain = np.zeros(iterations)

    for i in range(iterations):
        # Generate candidate jump from log-normal distribution
        epsilon = np.random.normal(0, np.sqrt(sigma_sq))
        lam_p = lam * np.exp(epsilon)

        # Calculate acceptance probability with asymmetric correction
        log_A = (alpha + sum_x) * np.log(lam_p / lam) - (beta + n_data) * (lam_p - lam)

        # Evaluate acceptance criterion
        if np.log(np.random.rand()) < log_A:
            lam = lam_p
            accepts += 1

        # Record current position in chain
        chain[i] = lam

    return accepts / iterations, chain


print("\n--- Part 2b: Sigma Trial ---")

# Search for proposal variance whose acceptance rate is closest to the midpoint
# of the target window [0.25, 0.40]
target_midpoint = (0.25 + 0.40) / 2
best_sigma = None
best_rate = None
sample_mean = sum_x / n_data

for s_sq in np.arange(0.05, 1.05, 0.05):
    rate, _ = run_mh(s_sq, 2000, start_val=sample_mean)
    if 0.25 <= rate <= 0.40:
        if best_sigma is None or abs(rate - target_midpoint) < abs(best_rate - target_midpoint):
            best_sigma = s_sq
            best_rate = rate

if best_sigma is not None:
    print(f"sigma_sq_trial: {best_sigma:.2f}")
    print(f"rate_trial: {best_rate:.4f}")

# Execute full chain using optimized variance
if best_sigma is not None:
    final_rate, _ = run_mh(best_sigma, 10000, start_val=sample_mean)
    print(f"sigma_sq_final: {best_sigma:.2f}")
    print(f"rate_final: {final_rate:.4f}")
else:
    print("Error: No optimal variance found within range.")

print("\n--- Part 3a: Independent Chains ---")

# Setup configuration for multiple parallel chains
chain_length = 3000
num_chains = 4
start_values = [0.5, 2.0, 5.0, 10.0]

# Generate independent sequences
raw_chains = []
for i in range(num_chains):
    _, full_chain = run_mh(best_sigma, chain_length, start_val=start_values[i])
    raw_chains.append(full_chain)
    print(f"Chain {i + 1} generated starting at {start_values[i]:.1f}")

raw_chains = np.array(raw_chains)

print("\n--- Part 3b: Burn-in ---")

# Discard initial iterations to remove startup location bias
burn_in = 1000
burned_chains = raw_chains[:, burn_in:]

print(f"Discarded first {burn_in} samples from each chain.")
print(f"Remaining active samples per chain: {burned_chains.shape[1]}")

print("\n--- Part 3c: Diagnostics ---")

# Display combined traceplot
plt.figure(figsize=(10, 5))
for i in range(num_chains):
    plt.plot(burned_chains[i], alpha=0.7, label=f'Chain {i + 1}')
plt.title('Traceplots of MCMC chains')
plt.xlabel('Iteration (Post Burn-in)')
plt.ylabel('Lambda', labelpad=15)
plt.legend()
plt.grid(True)
plt.show()

# Package data for arviz compatibility
mcmc_data = az.from_dict({"posterior": {"lambda": burned_chains}})

# Render autocorrelation functions via arviz
az.plot_autocorr(mcmc_data)

# Safely grab the current active axis drawn by ArviZ
ax = plt.gca()

# Clean up the labels and title
ax.set_title('')
ax.set_ylabel('Autocorrelation')

# Add the legend
lines = [line for line in ax.get_lines() if line.get_linestyle() != '--'][:num_chains]
labels = [f'Chain {i + 1}' for i in range(num_chains)]
ax.legend(lines, labels, loc='upper right')

# Apply final sizing and show
plt.gcf().set_size_inches(10, 5)
plt.suptitle('Autocorrelations of MCMC chains', fontsize=14)
plt.show()


# Helper function to compute effective sample size via Fast Fourier Transform
def compute_ess(chain):
    n = len(chain)
    chain_centered = chain - np.mean(chain)

    # Calculate normalized autocorrelation
    acf_full = np.fft.irfft(np.abs(np.fft.rfft(chain_centered, n=2 * n)) ** 2)
    acf = acf_full[:n] / acf_full[0]

    # Sum autocorrelations until encountering negative noise
    ess_sum = 1 + 2 * np.sum(acf[1:][acf[1:] > 0])
    return n / ess_sum


print("\nEffective Sample Size (ESS) per chain:")
total_ess = 0
for i in range(num_chains):
    # Compute and aggregate independent efficiency metrics
    ess = compute_ess(burned_chains[i])
    total_ess += ess
    print(f"Chain {i + 1} ESS: {ess:.2f}")

print(f"\nTotal ESS (RWMH): {total_ess:.2f}")

print("\n--- Part 4a: Independence Sampler Chains ---")

# Define proposal parameters to match posterior mean
# Posterior is Gamma(20, 7) so we use shape=20 and scale=1/7
k_prop = 20
theta_prop = 1 / 7


# Independence Metropolis-Hastings transition kernel
def run_independence_mh(iterations, start_val):
    lam = start_val
    accepts = 0
    chain = np.zeros(iterations)

    for i in range(iterations):
        # Draw proposal independently from Gamma distribution
        lam_p = np.random.gamma(k_prop, theta_prop)

        # Target log-density for proposed and current values
        log_target_p = 19 * np.log(lam_p) - 7 * lam_p
        log_target_curr = 19 * np.log(lam) - 7 * lam

        # Proposal log-density for proposed and current values
        log_prop_p = (k_prop - 1) * np.log(lam_p) - (lam_p / theta_prop)
        log_prop_curr = (k_prop - 1) * np.log(lam) - (lam / theta_prop)

        # Calculate independence sampler acceptance ratio
        log_A = (log_target_p - log_prop_p) - (log_target_curr - log_prop_curr)

        # Evaluate acceptance criterion
        if np.log(np.random.rand()) < log_A:
            lam = lam_p
            accepts += 1

        # Record current position in chain
        chain[i] = lam

    return accepts / iterations, chain

# Generate independent sequences using the new sampler
ind_raw_chains = []
for i in range(num_chains):
    rate, full_chain = run_independence_mh(chain_length, start_val=start_values[i])
    ind_raw_chains.append(full_chain)
    print(f"Chain {i + 1} generated starting at {start_values[i]:.1f}")

ind_raw_chains = np.array(ind_raw_chains)

print("\n--- Part 4b: Burn-in ---")

# Discard initial iterations
ind_burned_chains = ind_raw_chains[:, burn_in:]

print(f"Discarded first {burn_in} samples from each chain.")
print(f"Remaining active samples per chain: {ind_burned_chains.shape[1]}")

print("\n--- Part 4c: Diagnostics ---")

# Display combined traceplot for independence sampler
plt.figure(figsize=(10, 5))
for i in range(num_chains):
    plt.plot(ind_burned_chains[i], alpha=0.7, label=f'Chain {i + 1}')
plt.title('Traceplots of Independence MCMC chains')
plt.xlabel('Iteration (Post Burn-in)')
plt.ylabel('Lambda', labelpad=15)
plt.legend()
plt.grid(True)
plt.show()

# Package data for arviz compatibility
mcmc_data_ind = az.from_dict({"posterior": {"lambda": ind_burned_chains}})

# Render autocorrelation functions via arviz
az.plot_autocorr(mcmc_data_ind)

# Safely grab the current active axis drawn by ArviZ
ax = plt.gca()

# Clean up the labels and title
ax.set_title('')
ax.set_ylabel('Autocorrelation')

# Add the legend
lines = [line for line in ax.get_lines() if line.get_linestyle() != '--'][:num_chains]
labels = [f'Chain {i + 1}' for i in range(num_chains)]
ax.legend(lines, labels, loc='upper right')

# Apply final sizing and show
plt.gcf().set_size_inches(10, 5)
plt.suptitle('Autocorrelations of Independence chains', fontsize=14)
plt.show()

print("\nEffective Sample Size (ESS) per chain:")
ind_total_ess = 0
for i in range(num_chains):
    # Compute and aggregate independent efficiency metrics
    ess = compute_ess(ind_burned_chains[i])
    ind_total_ess += ess
    print(f"Chain {i + 1} ESS: {ess:.2f}")

print(f"\nTotal ESS (Independence Sampler): {ind_total_ess:.2f}")
