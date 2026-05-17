"""
Mathematics 2 - Homework GD.1
Programming Problems 5 and 6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# PROBLEM 5: Projected Gradient Descent
# ============================================================

# --- Function and gradient from Problem 2 ---

def f(point):
    """f(x,y) = x^2 + e^x + y^2 - xy from Problem 2."""
    x, y = point
    return x**2 + np.exp(x) + y**2 - x*y

def grad_f(point):
    """Gradient of f at (x,y): (2x + e^x - y, 2y - x)."""
    x, y = point
    return np.array([2*x + np.exp(x) - y, 2*y - x])

# --- Constants from Problem 2 ---

L     = np.sqrt((6 + np.exp(2))**2 + 36)                  # Lipschitz constant: max ||grad f|| on K
beta  = ((4 + np.exp(2)) + np.sqrt(np.exp(4) + 4)) / 2    # smoothness: max eigenvalue of Hessian on K
alpha = ((4 + np.exp(-2)) - np.sqrt(np.exp(-4) + 4)) / 2  # strong convexity: min eigenvalue of Hessian on K
kappa = beta / alpha                                        # condition number

# find x* numerically since f is strictly convex it has a unique minimum
x_star = minimize(f, [0.0, 0.0]).x
f_star = f(x_star)

print("=" * 60)
print("PROBLEM 5: Projected Gradient Descent")
print("=" * 60)
print(f"\nConstants from Problem 2:")
print(f"  L     = {L:.4f}  (Lipschitz constant)")
print(f"  beta  = {beta:.4f}  (smoothness)")
print(f"  alpha = {alpha:.4f}  (strong convexity)")
print(f"  kappa = {kappa:.4f}  (condition number beta/alpha)")
print(f"\nUnconstrained minimum (found numerically):")
print(f"  x*    = ({x_star[0]:.4f}, {x_star[1]:.4f})")
print(f"  f(x*) = {f_star:.6f}")

# --- Projection functions from Problem 3 ---

def project_disk(point):
    """
    Project onto disk x^2 + y^2 <= 1.5.
    If inside the disk return the point unchanged.
    If outside scale it radially onto the boundary circle.
    """
    radius = np.sqrt(1.5)
    norm = np.linalg.norm(point)
    if norm <= radius:
        return point
    return radius * point / norm

def project_square(point):
    """
    Project onto square [-1,1] x [-1,1].
    Each coordinate is projected independently by clamping to [-1, 1].
    """
    return np.clip(point, -1, 1)

def project_onto_edge(point, U, V):
    """
    Project point onto edge from U to V.
    Compute the foot of the perpendicular onto the infinite line through U and V,
    then clamp t to [0,1] so the result stays on the edge and not beyond the vertices.
    """
    edge = V - U
    t = np.dot(point - U, edge) / np.dot(edge, edge)
    return U + np.clip(t, 0, 1) * edge

def project_triangle(point):
    """
    Project onto triangle A=(-1,-1), B=(1.5,-1), C=(-1,1.5).
    Inside iff: y>=-1 (above AB), x>=-1 (right of AC), x+y<=0.5 (below BC).
    One violation -> project onto that edge. Two violations -> snap to shared vertex.
    """
    x, y = point
    A = np.array([-1.0, -1.0])
    B = np.array([ 1.5, -1.0])
    C = np.array([-1.0,  1.5])

    below_AB = y < -1
    left_AC  = x < -1
    above_BC = x + y > 0.5

    if not below_AB and not left_AC and not above_BC:
        return point

    if below_AB and left_AC:
        return A
    if below_AB and above_BC:
        return B
    if left_AC and above_BC:
        return C

    if below_AB:
        return project_onto_edge(point, A, B)
    if left_AC:
        return project_onto_edge(point, A, C)
    return project_onto_edge(point, B, C)

# --- PGD loop (shared between problems 5 and 6) ---

def run_pgd(starting_point, projection, gamma, n_steps, objective, gradient):
    """
    Run Projected Gradient Descent for n_steps steps.
    Returns the final point and list of objective values at each step.
    """
    x = starting_point.copy()
    objective_values = [objective(x)]

    for _ in range(n_steps):
        x = projection(x - gamma * gradient(x))
        objective_values.append(objective(x))

    return x, objective_values

# --- Theoretical bounds from Theorem 3.3 ---

def bound_case2(step, R, beta, f_at_x1, f_star):
    """Theorem 3.3 case 2 bound: (3*beta*R^2 + f(x1) - f(x*)) / step."""
    return (3 * beta * R**2 + f_at_x1 - f_star) / max(step, 1)

def bound_case3(step, R, beta, kappa):
    """Theorem 3.3 case 3 bound: beta/2 * ((kappa-1)/kappa)^(2*step) * R^2."""
    return (beta / 2) * ((kappa - 1) / kappa)**(2 * step) * R**2

# --- Learning rates from Theorem 3.3 ---

x1      = np.array([-1.0, 1.0])   # starting point given in problem
n_steps = 10                       # number of PGD steps
f_at_x1 = f(x1)                    # objective value at starting point

# distance from starting point to the true minimum, needed for case 1 learning rate
R = np.linalg.norm(x1 - x_star)

# case 1 (L-Lipschitz, Theorem 3.3 case 1): gamma = R / (L * sqrt(T))
gamma_case1 = R / (L * np.sqrt(n_steps))

# case 2 (beta-smooth, Theorem 3.3 case 2): gamma = 1/beta
gamma_case2 = 1.0 / beta

# case 3 (alpha-strongly convex + beta-smooth, Theorem 3.3 case 3): gamma = 2/(alpha+beta)
gamma_case3 = 2.0 / (alpha + beta)

learning_rates = {
    "Case 1 (L-Lipschitz)":        (gamma_case1, None),
    "Case 2 (beta-smooth)":        (gamma_case2, "case2"),
    "Case 3 (alpha + beta smooth)": (gamma_case3, "case3"),
}

projections = {
    "Disk (x^2+y^2 <= 1.5)":               project_disk,
    "Square [-1,1] x [-1,1]":               project_square,
    "Triangle ((-1,-1), (1.5,-1), (-1,1.5))": project_triangle,
}

print(f"\nStarting point: x1 = {x1}")
print(f"Distance to x*: R = ||x1 - x*|| = {R:.4f}")
print(f"Number of steps: {n_steps}")
print(f"\nLearning rates from Theorem 3.3:")
print(f"  Case 1: gamma = R / (L * sqrt(T)) = {gamma_case1:.4f}")
print(f"  Case 2: gamma = 1 / beta          = {gamma_case2:.4f}")
print(f"  Case 3: gamma = 2 / (alpha + beta)= {gamma_case3:.4f}")

# --- Run all 9 combinations ---

results = {}
for proj_name, projection in projections.items():
    for lr_name, (gamma, _) in learning_rates.items():
        final_x, objective_values = run_pgd(x1, projection, gamma, n_steps, f, grad_f)
        results[(proj_name, lr_name)] = {
            "objective_values": objective_values,
            "final_x":          final_x,
            "final_f":          objective_values[-1],
        }

# --- Print results ---

print("\n" + "-" * 60)
print(f"Results after {n_steps} steps (f(x*) = {f_star:.6f}):")
print("-" * 60)

for proj_name in projections:
    print(f"\nDomain: {proj_name}")
    for lr_name, (gamma, bound_type) in learning_rates.items():
        r   = results[(proj_name, lr_name)]
        difference = r["final_f"] - f_star

        if bound_type == "case2":
            bound = bound_case2(n_steps, R, beta, f_at_x1, f_star)
            bound_str = f"Theorem 3.3 case 2 bound = {bound:.4f}, satisfied = {difference <= bound}"
        elif bound_type == "case3":
            bound = bound_case3(n_steps, R, beta, kappa)
            bound_str = f"Theorem 3.3 case 3 bound = {bound:.4f}, satisfied = {difference <= bound}"
        else:
            bound_str = "no bound computed for case 1"

        print(f"  {lr_name} (gamma = {gamma:.4f})")
        print(f"    Final x    = ({r['final_x'][0]:.6f}, {r['final_x'][1]:.6f})")
        print(f"    Final f(x) = {r['final_f']:.6f}  (difference = {difference:.6f}, {bound_str})")

print("\n" + "-" * 60)
print("Best performer per domain (lowest f after 10 steps):")
print("-" * 60)
for proj_name in projections:
    best_lr = min(learning_rates, key=lambda lr: results[(proj_name, lr)]["final_f"])
    best_f  = results[(proj_name, best_lr)]["final_f"]
    print(f"  {proj_name}: {best_lr} (f = {best_f:.6f})")

# --- Plot convergence: one subplot per domain ---
# colors = learning rates, dashed lines = Theorem 3.3 bounds

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Problem 5: PGD convergence, f(x_k) vs theoretical bounds", fontsize=13)

steps   = np.arange(n_steps + 1)
bounds2 = [bound_case2(step, R, beta, f_at_x1, f_star) + f_star for step in steps]
bounds3 = [bound_case3(step, R, beta, kappa) + f_star for step in steps]

for ax, proj_name in zip(axes, projections):
    colors = ["tab:green", "tab:orange", "tab:blue"]
    for (lr_name, (gamma, _)), color in zip(learning_rates.items(), colors):
        objective_values = results[(proj_name, lr_name)]["objective_values"]
        ax.plot(steps, objective_values, marker="o", markersize=4, color=color, label=lr_name)

    # case 1 bound is on the average of iterates, not f(x_k) directly, so not plotted here
    ax.plot(steps, bounds2, "k--", linewidth=1, label="Theorem 3.3 case 2 bound")
    ax.plot(steps, bounds3, "k:",  linewidth=1, label="Theorem 3.3 case 3 bound")
    ax.axhline(f_star, color="gray", linewidth=1, label=f"f(x*) = {f_star:.3f}")

    ax.set_title(proj_name)
    ax.set_xlabel("Step k")
    ax.set_ylabel("f(x_k)")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("problem5_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to problem5_convergence.png")

# ============================================================
# PROBLEM 6: Hartmann 3D Function
# ============================================================

print("\n" + "=" * 60)
print("PROBLEM 6: Hartmann 3D Function")
print("=" * 60)

# --- Function parameters from the problem table ---

# each row i corresponds to one of 4 Gaussian terms in the sum
A_hartmann = np.array([   # a_ij: width coefficient for term i in dimension j
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
])

c_hartmann = np.array([1.0, 1.2, 3.0, 3.2])   # c_i: weight of term i

P_hartmann = np.array([   # p_ij: center of term i in dimension j
    [0.36890, 0.11700, 0.26730],
    [0.46990, 0.43870, 0.74700],
    [0.10910, 0.87320, 0.55470],
    [0.03815, 0.57430, 0.88280],
])

KNOWN_MINIMUM = -3.86278214782076

# --- Hartmann function and gradient ---

def hartmann3(z):
    """Hartmann 3D function: sum of four weighted Gaussians, negated for minimization."""
    total = 0.0
    for i in range(4):
        total += c_hartmann[i] * np.exp(-np.sum(A_hartmann[i] * (z - P_hartmann[i])**2))
    return -total

def grad_hartmann3(z):
    """
    Gradient of Hartmann 3D via chain rule on each Gaussian term.

    For each term i, let gauss_i = exp(-sum_j a_ij (z_j - p_ij)^2). Then:
        d(gauss_i)/d(z_k) = gauss_i * (-2 * a_ik * (z_k - p_ik))

    Since f = -sum_i c_i * gauss_i, the full gradient is:
        df/dz = sum_i c_i * gauss_i * 2 * A_i * (z - p_i)
    """
    grad = np.zeros(3)
    for i in range(4):
        diff = z - P_hartmann[i]
        gauss_i = np.exp(-np.sum(A_hartmann[i] * diff**2))
        grad += c_hartmann[i] * gauss_i * 2 * A_hartmann[i] * diff
    return grad

def project_unit_cube(z):
    """Project onto [0,1]^3 by clamping each coordinate independently."""
    return np.clip(z, 0, 1)

# --- Run PGD ---

z1               = np.array([0.5, 0.5, 0.5])   # center of domain as starting point
n_steps_hartmann = 10000
gamma_hartmann   = 0.001                        # standard small learning rate

print(f"\nStarting point: z1 = {z1}")
print(f"Learning rate:  gamma = {gamma_hartmann}")
print(f"Number of steps: {n_steps_hartmann}")
print(f"Known minimum: {KNOWN_MINIMUM}")

z_final, objective_values_hartmann = run_pgd(
    z1, project_unit_cube, gamma_hartmann,
    n_steps_hartmann, hartmann3, grad_hartmann3
)

difference = abs(objective_values_hartmann[-1] - KNOWN_MINIMUM)
steps_to_near_minimum = next(
    (step for step, fv in enumerate(objective_values_hartmann)
     if abs(fv - KNOWN_MINIMUM) < 0.01), None
)

print(f"\nResults:")
print(f"  Final point         = ({z_final[0]:.6f}, {z_final[1]:.6f}, {z_final[2]:.6f})")
print(f"  f at final point    = {objective_values_hartmann[-1]:.10f}")
print(f"  Known minimum       = {KNOWN_MINIMUM:.10f}")
print(f"  Difference after {n_steps_hartmann} steps = {difference:.2e}")
print(f"  Steps to get within 0.01 of minimum = {steps_to_near_minimum}")

# --- Plot convergence ---

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Problem 6: Hartmann 3D, PGD convergence", fontsize=13)

all_steps = np.arange(len(objective_values_hartmann))

ax.plot(all_steps, objective_values_hartmann, linewidth=1, label="f(z_k)")
ax.axhline(KNOWN_MINIMUM, color="red", linestyle="--", label=f"Known minimum {KNOWN_MINIMUM:.4f}")
ax.set_xlabel("Step k")
ax.set_ylabel("f(z_k)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("problem6_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to problem6_convergence.png")
