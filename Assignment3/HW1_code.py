"""
Mathematics 2 - Homework GD.1

Problem 5: PGD on f(x,y) = x^2 + e^x + y^2 - xy
  9 combinations: 3 domains x 3 learning rates from Theorem 3.3

Problem 6: Minimize Hartmann-3 function on [0, 1]^3 using PGD
"""

import numpy as np


# ---------- Problem 5 ----------

# Function and gradient from Problem 2
def f(p):
    x, y = p
    return x**2 + np.exp(x) + y**2 - x*y


def grad_f(p):
    x, y = p
    return np.array([2*x + np.exp(x) - y, 2*y - x])


# Constants from Problem 2 (computed analytically on K = [-2, 2] x [-2, 2])
ALPHA = ((4 + np.exp(-2)) - np.sqrt(np.exp(-4) + 4)) / 2  # ~1.065, strong convexity
BETA  = ((4 + np.exp(2))  + np.sqrt(np.exp(4)  + 4)) / 2  # ~9.522, smoothness
L     = np.sqrt((6 + np.exp(2))**2 + 36)                  # ~14.67, Lipschitz constant


# Projections from Problem 3
def project_disk(p, r=np.sqrt(1.5)):
    # If inside, return as is. Otherwise scale down to the boundary.
    norm = np.linalg.norm(p)
    if norm <= r:
        return p.copy()
    return r * p / norm


def project_square(p, a=-1, b=1):
    # Each coordinate is projected independently onto [a, b]
    return np.clip(p, a, b)


def project_triangle(p):
    # Triangle defined by x >= -1, y >= -1, x + y <= 0.5
    p1, p2 = p
    if p1 >= -1 and p2 >= -1 and p1 + p2 <= 0.5:
        return p.copy()

    # Project onto each edge and return the closest result
    candidates = [
        np.array([np.clip(p1, -1, 1.5), -1.0]),            # bottom edge
        np.array([-1.0, np.clip(p2, -1, 1.5)]),             # left edge
    ]

    # Hypotenuse x + y = 0.5: project onto line then clamp to segment BC
    qx, qy = (p1 - p2 + 0.5) / 2, (p2 - p1 + 0.5) / 2
    if qx < -1:
        candidates.append(np.array([-1.0, 1.5]))            # vertex C
    elif qx > 1.5:
        candidates.append(np.array([1.5, -1.0]))            # vertex B
    else:
        candidates.append(np.array([qx, qy]))

    return min(candidates, key=lambda q: np.linalg.norm(p - q))


# PGD: x_{k+1} = pi_D(x_k - gamma * grad_f(x_k))
def pgd(x1, gamma, project, n_steps=10):
    xs = [np.array(x1, dtype=float)]
    for _ in range(n_steps):
        x_new = project(xs[-1] - gamma * grad_f(xs[-1]))
        xs.append(x_new)
    return xs


# Find x* on a domain by running PGD to convergence with gamma = 1/beta
def find_min(project, x0=np.array([0.0, 0.0])):
    x = x0.copy()
    for _ in range(100000):
        x_new = project(x - (1 / BETA) * grad_f(x))
        if np.linalg.norm(x_new - x) < 1e-14:
            break
        x = x_new
    return x


def problem5():
    x1 = np.array([-1.0, 1.0])
    T = 10
    kappa = BETA / ALPHA

    domains = {"Disk": project_disk, "Square": project_square, "Triangle": project_triangle}

    print("Problem 5: PGD on f(x,y) = x^2 + e^x + y^2 - xy")
    print(f"Constants: alpha = {ALPHA:.4f}, beta = {BETA:.4f}, L = {L:.4f}, kappa = {kappa:.4f}")
    print()

    print(f"{'Domain':<10} {'Rate':<27} {'gamma':>8} {'gap f(x) - f(x*)':>18} {'bound':>12}")
    print("-" * 78)

    for dom_name, project in domains.items():
        x_star = find_min(project)
        f_star = f(x_star)
        R = np.linalg.norm(x1 - x_star)

        # Three learning rates and bounds from Theorem 3.3
        # Rate 1 (L-Lipschitz):       gamma = R/(L*sqrt(T)), bound on f(avg iterates) - f*
        # Rate 2 (beta-smooth):       gamma = 1/beta,        bound on f(x_T) - f*
        # Rate 3 (alpha-SC + smooth): gamma = 1/beta,        tighter bound using kappa
        gamma_1 = R / (L * np.sqrt(T))
        gamma_2 = 1 / BETA
        gamma_3 = 1 / BETA  # same as gamma_2, but bound differs due to strong convexity
        bound_1 = L * R / np.sqrt(T)
        bound_2 = (3 * BETA * R**2 + f(x1) - f_star) / T
        bound_3 = (BETA / 2) * ((kappa - 1) / kappa)**(2 * T) * R**2

        # Rate 1 uses the average iterate as stated in Theorem 3.3
        gap_1 = f(np.mean(pgd(x1, gamma_1, project, T)[1:], axis=0)) - f_star

        # Rates 2 and 3 use the last iterate
        # They share the same gamma and trajectory, only the bound differs
        gap_2 = f(pgd(x1, gamma_2, project, T)[-1]) - f_star
        gap_3 = f(pgd(x1, gamma_3, project, T)[-1]) - f_star

        print(f"{dom_name:<10} {'(1) L-Lipschitz':<27} {gamma_1:>8.4f} {gap_1:>18.6f} {bound_1:>12.4f}")
        print(f"{dom_name:<10} {'(2) beta-smooth':<27} {gamma_2:>8.4f} {gap_2:>18.6f} {bound_2:>12.4f}")
        print(f"{dom_name:<10} {'(3) alpha-SC + smooth':<27} {gamma_3:>8.4f} {gap_3:>18.6f} {bound_3:>12.4f}")
        print()


# ---------- Problem 6 ----------

# Hartmann-3 function on [0, 1]^3
# f(z) = -sum_i c_i * exp(-sum_j a_{i,j} * (z_j - p_{i,j})^2)
# Parameters from the assignment table:

A_HART = np.array([
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
])
C_HART = np.array([1.0, 1.2, 3.0, 3.2])
P_HART = np.array([
    [0.36890, 0.11700, 0.26730],
    [0.46990, 0.43870, 0.74700],
    [0.10910, 0.87320, 0.55470],
    [0.03815, 0.57430, 0.88280],
])

HART_STAR = -3.86278214782076


def hartmann(z):
    diff = z - P_HART
    inner = np.sum(A_HART * diff**2, axis=1)
    return -np.sum(C_HART * np.exp(-inner))


def grad_hartmann(z):
    # df/dz_k = sum_i 2 * c_i * a_{i,k} * (z_k - p_{i,k}) * exp(-inner_i)
    diff = z - P_HART
    exps = np.exp(-np.sum(A_HART * diff**2, axis=1))
    return np.sum((2 * C_HART * exps)[:, None] * A_HART * diff, axis=0)


def problem6():
    z = np.array([0.5, 0.5, 0.5])
    gamma = 0.01
    n_steps = 5000

    print("Problem 6: Minimizing Hartmann-3 on [0, 1]^3")
    print(f"Starting point: {z},  f = {hartmann(z):.6f},  target f* = {HART_STAR}")
    print(f"Learning rate: gamma = {gamma}")
    print()

    # PGD with box projection onto [0, 1]^3 (project_square with a=0, b=1)
    for k in range(n_steps):
        z_new = project_square(z - gamma * grad_hartmann(z), 0, 1)
        if np.linalg.norm(z_new - z) < 1e-14:
            break
        z = z_new

    print(f"After {k + 1} steps:")
    print(f"  z   = ({z[0]:.6f}, {z[1]:.6f}, {z[2]:.6f})")
    print(f"  f   = {hartmann(z):.15f}")
    print(f"  gap = {hartmann(z) - HART_STAR:.2e}")


if __name__ == "__main__":
    problem5()
    problem6()
