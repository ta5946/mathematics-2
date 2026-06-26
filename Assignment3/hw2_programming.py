"""
Mathematics 2 - Homework GD.2
Implementation of all optimisation methods:
    GD, Polyak GD, Nesterov GD, AdaGrad, Adam, Newton, BFGS
"""

import numpy as np
import time


# ============================================================
# FIRST-ORDER METHODS
# ============================================================

def run_gd(x1, grad, f, n_steps, gamma=0.001):
    """
    Gradient Descent: x_{k+1} = x_k - gamma * grad_f(x_k).

    Parameters
    ----------
    x1      : starting point (numpy array)
    grad    : gradient function
    f       : objective function, used only to record history
    n_steps : number of steps
    gamma   : learning rate

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    f_history = [f(x)]

    for _ in range(n_steps):
        x = x - gamma * grad(x)
        f_history.append(f(x))

    return x, f_history


def run_polyak_gd(x1, grad, f, n_steps, gamma=0.001, mu=0.9):
    """
    Polyak GD (Heavy Ball): x_{k+1} = x_k - gamma * grad_f(x_k) + mu * (x_k - x_{k-1}).

    Momentum term mu * (x_k - x_{k-1}) dampens oscillation in directions where
    the gradient alternates sign and accelerates progress where it is consistent.
    First step uses x_0 = x_1 convention so there is no momentum on step one.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    gamma   : gradient step size
    mu      : momentum coefficient

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    x_prev    = x1.copy()   # x_0 = x_1: no momentum on first step
    f_history = [f(x)]

    for _ in range(n_steps):
        x_next = x - gamma * grad(x) + mu * (x - x_prev)
        x_prev = x
        x      = x_next
        f_history.append(f(x))

    return x, f_history


def run_nesterov_gd(x1, grad, f, n_steps, gamma=0.001, mu=0.9):
    """
    Nesterov GD: momentum applied first to form a lookahead point, gradient step taken from there.

        y_{k}   = x_k + mu * (x_k - x_{k-1})    (lookahead)
        x_{k+1} = y_k - gamma * grad_f(y_k)

    Reordering momentum and gradient step compared to Polyak GD gives provable
    convergence on general convex functions with an O(1/k^2) rate.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    gamma   : gradient step size
    mu      : momentum coefficient

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    x_prev    = x1.copy()   # x_0 = x_1 convention
    f_history = [f(x)]

    for _ in range(n_steps):
        lookahead = x + mu * (x - x_prev)
        x_next    = lookahead - gamma * grad(lookahead)
        x_prev    = x
        x         = x_next
        f_history.append(f(x))

    return x, f_history


def run_adagrad(x1, grad, f, n_steps, gamma=0.01, eps=1e-8):
    """
    AdaGrad: per-coordinate learning rate scaled by cumulative squared gradients.

        sq_grad_sum_i += grad_f(x_k)_i^2
        x_{k+1,i}     = x_{k,i} - gamma / sqrt(sq_grad_sum_i + eps) * grad_f(x_k)_i

    Coordinates with historically large gradients are dampened; quiet ones keep
    a large effective step size. Note: sq_grad_sum only grows, so the step size
    decays to zero permanently. This is AdaGrad's known limitation.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    gamma   : global learning rate
    eps     : added to denominator to avoid division by zero (default 1e-8)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x           = x1.copy()
    sq_grad_sum = np.zeros_like(x)   # cumulative sum of squared gradients per coordinate
    f_history   = [f(x)]

    for _ in range(n_steps):
        grad_x      = grad(x)
        sq_grad_sum = sq_grad_sum + grad_x**2
        x           = x - gamma / np.sqrt(sq_grad_sum + eps) * grad_x
        f_history.append(f(x))

    return x, f_history


def run_adam(x1, grad, f, n_steps,
             gamma=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam (Adaptive Moment Estimation): momentum + per-coordinate adaptive scaling.

    Maintains two decaying averages per coordinate:
        m_k : smoothed gradient          (direction / momentum)
        v_k : smoothed squared gradient  (per-coordinate scale)

        m_k     = beta1 * m_{k-1} + (1 - beta1) * grad_f(x_k)
        v_k     = beta2 * v_{k-1} + (1 - beta2) * grad_f(x_k)^2
        m_hat_k = m_k / (1 - beta1^k)    (bias correction)
        v_hat_k = v_k / (1 - beta2^k)    (bias correction)
        x_{k+1} = x_k - gamma * m_hat_k / (sqrt(v_hat_k) + eps)

    Bias correction compensates for m and v being zero-initialised, which would
    otherwise severely underestimate both quantities in early steps.
    Unlike AdaGrad, decaying averages prevent the step size from collapsing to zero.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    gamma   : global learning rate        (default 0.001)
    beta1   : first moment decay rate     (default 0.9)
    beta2   : second moment decay rate    (default 0.999)
    eps     : numerical stability constant (default 1e-8)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    m         = np.zeros_like(x)   # first moment  (momentum)
    v         = np.zeros_like(x)   # second moment (adaptive scale)
    f_history = [f(x)]

    for step in range(1, n_steps + 1):
        grad_x = grad(x)

        m = beta1 * m + (1 - beta1) * grad_x       # first moment update
        v = beta2 * v + (1 - beta2) * grad_x**2    # second moment update

        m_hat = m / (1 - beta1**step)              # bias-corrected first moment
        v_hat = v / (1 - beta2**step)              # bias-corrected second moment

        x = x - gamma * m_hat / (np.sqrt(v_hat) + eps)
        f_history.append(f(x))

    return x, f_history


# ============================================================
# SECOND-ORDER METHODS
# ============================================================

def run_newton(x1, grad, hess, f, n_steps, alpha=1.0, lam=0.0):
    """
    Newton Method: x_{k+1} = x_k - alpha * H_k^{-1} * grad_f(x_k).

    H_k = hess_f(x_k) + lam * I (Hessian, optionally regularised).
    Uses np.linalg.solve instead of explicit inversion for numerical stability.
    Raises ValueError if H_k is singular (use lam > 0 to regularise).

    alpha : step scale; reduce below 1.0 to avoid overshooting (line search).
    lam   : shifts all Hessian eigenvalues up by lam, enforcing positive
            definiteness when the Hessian is indefinite. Large lam resembles GD.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function, returns array of shape (n,)
    hess    : Hessian function,  returns array of shape (n, n)
    f       : objective function
    n_steps : number of steps
    alpha   : step scale (default 1.0)
    lam     : Hessian regularisation (default 0.0)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    n_dim     = len(x)
    f_history = [f(x)]

    for _ in range(n_steps):
        grad_x = grad(x)
        H      = hess(x) + lam * np.eye(n_dim)

        try:
            direction = np.linalg.solve(H, grad_x)   # H * direction = grad_x
        except np.linalg.LinAlgError:
            raise ValueError(
                "Newton: Hessian is singular at the current iterate. "
                "Use lam > 0 to regularise."
            )

        x = x - alpha * direction
        f_history.append(f(x))

    return x, f_history


def run_bfgs(x1, grad, f, n_steps, eps=1e-10, c=1e-4, tau=0.5):
    """
    BFGS: quasi-Newton method with a rank-2 inverse Hessian approximation
    and backtracking line search (Armijo condition).

    Maintains B_k approximating (hess_f)^{-1}, updated each step from:
        delta_k = x_k - x_{k-1}                      (step taken)
        y_k     = grad_f(x_k) - grad_f(x_{k-1})      (gradient change)

    Rank-2 update (compact form):
        rho      = 1 / (delta^T y)
        V_k      = I - rho * delta * y^T
        B_{k+1}  = V_k B_k V_k^T + rho * delta * delta^T

    Line search (backtracking Armijo):
        Start with alpha = 1.0, halve until:
        f(x_k + alpha * d_k) <= f(x_k) + c * alpha * grad_f(x_k)^T d_k
        This guarantees each step decreases f, keeping B_k well-behaved.

    Rank-2 update is skipped when delta^T y < eps to avoid numerical blow-up.
    B_1 is initialised to the identity, making the first step equivalent to GD.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    eps     : minimum value of delta^T y to apply rank-2 update (default 1e-10)
    c       : Armijo sufficient decrease constant (default 1e-4)
    tau     : backtracking reduction factor per line search iteration (default 0.5)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    n_dim     = len(x)
    B         = np.eye(n_dim)   # B_1 = I: identity as initial inverse Hessian approximation
    f_history = [f(x)]
    grad_x    = None            # evaluated lazily inside the loop

    for _ in range(n_steps):
        if grad_x is None:
            grad_x = grad(x)

        # descent direction
        direction = B @ grad_x

        # backtracking line search: start at alpha=1 and halve until Armijo holds
        alpha     = 1.0
        f_x       = f(x)
        slope     = grad_x @ direction   # grad_f(x)^T * direction, should be > 0
        while True:
            x_candidate = x - alpha * direction
            f_candidate = f(x_candidate)
            if np.isfinite(f_candidate) and f_candidate <= f_x - c * alpha * slope:
                break
            alpha *= tau
            if alpha < 1e-16:   # line search failed; stay put
                break

        x_next    = x - alpha * direction
        grad_next = grad(x_next)

        delta       = x_next - x
        grad_change = grad_next - grad_x

        delta_dot_y = delta @ grad_change
        if delta_dot_y > eps:
            rho = 1.0 / delta_dot_y
            V_k = np.eye(n_dim) - rho * np.outer(delta, grad_change)
            B   = V_k @ B @ V_k.T + rho * np.outer(delta, delta)

        x      = x_next
        grad_x = grad_next
        f_history.append(f(x))

    return x, f_history


# ============================================================
# PROBLEM 6: FUNCTIONS, GRADIENTS, HESSIANS, STARTING POINTS
# ============================================================

# ------------------------------------------------------------
# (a) f(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y
#
# This is a quadratic so the Hessian is constant.
# Let r1 = x-z, r2 = 2y+z, r3 = 4x-2y+z.
# ------------------------------------------------------------

def f_a(point):
    x, y, z = point
    r1 = x - z
    r2 = 2*y + z
    r3 = 4*x - 2*y + z
    return r1**2 + r2**2 + r3**2 + x + y

def grad_f_a(point):
    x, y, z = point
    r1 = x - z
    r2 = 2*y + z
    r3 = 4*x - 2*y + z
    df_dx = 2*r1 + 8*r3 + 1
    df_dy = 4*r2 - 4*r3 + 1
    df_dz = -2*r1 + 2*r2 + 2*r3
    return np.array([df_dx, df_dy, df_dz])

def hess_f_a(point):
    """
    Hessian is constant (f is quadratic).
    H_xx = 2*1^2 + 2*4^2 = 34
    H_yy = 2*2^2 + 2*2^2 = 16
    H_zz = 2*1^2 + 2*1^2 + 2*1^2 = 6
    H_xy = 2*1*0 + 2*4*(-2) = -16
    H_xz = 2*1*(-1) + 2*4*1 = 6
    H_yz = 2*2*1 + 2*(-2)*1 = 0
    """
    return np.array([
        [34,  -16,  6],
        [-16,  16,  0],
        [  6,   0,  6],
    ], dtype=float)

STARTING_POINTS_A = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
]

# ------------------------------------------------------------
# (b) f(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2
#
# Extended 3D Rosenbrock. Non-convex, ill-conditioned.
# Let u = y - x^2, v = z - y^2.
# ------------------------------------------------------------

def f_b(point):
    x, y, z = point
    return (x-1)**2 + (y-1)**2 + 100*(y - x**2)**2 + 100*(z - y**2)**2

def grad_f_b(point):
    x, y, z = point
    u = y - x**2
    v = z - y**2
    df_dx = 2*(x-1) - 400*x*u
    df_dy = 2*(y-1) + 200*u - 400*y*v
    df_dz = 200*v
    return np.array([df_dx, df_dy, df_dz])

def hess_f_b(point):
    """
    H_xx = 2 - 400*(y - x^2) + 800*x^2
    H_yy = 202 - 400*(z - y^2) + 800*y^2
    H_zz = 200
    H_xy = -400*x
    H_xz = 0
    H_yz = -400*y
    """
    x, y, z = point
    u = y - x**2
    v = z - y**2
    H_xx = 2 - 400*u + 800*x**2
    H_yy = 202 - 400*v + 800*y**2
    H_zz = 200.0
    H_xy = -400*x
    H_xz = 0.0
    H_yz = -400*y
    return np.array([
        [H_xx, H_xy, H_xz],
        [H_xy, H_yy, H_yz],
        [H_xz, H_yz, H_zz],
    ], dtype=float)

STARTING_POINTS_B = [
    np.array([1.2,  1.2, 1.2]),
    np.array([-1.0, 1.2, 1.2]),
]

# ------------------------------------------------------------
# (c) f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
#
# Beale function (3D generalisation). Non-convex.
# Let r1 = 1.5 - x + xy, r2 = 2.25 - x + xy^2, r3 = 2.625 - x + xy^3.
# ------------------------------------------------------------

def f_c(point):
    x, y = point
    r1 = 1.5   - x + x*y
    r2 = 2.25  - x + x*y**2
    r3 = 2.625 - x + x*y**3
    return r1**2 + r2**2 + r3**2

def grad_f_c(point):
    """
    df/dx = 2*r1*(y-1) + 2*r2*(y^2-1) + 2*r3*(y^3-1)
    df/dy = 2*r1*x + 2*r2*(2xy) + 2*r3*(3xy^2)
    """
    x, y = point
    r1 = 1.5   - x + x*y
    r2 = 2.25  - x + x*y**2
    r3 = 2.625 - x + x*y**3
    df_dx = 2*r1*(y-1) + 2*r2*(y**2-1) + 2*r3*(y**3-1)
    df_dy = 2*r1*x + 2*r2*(2*x*y) + 2*r3*(3*x*y**2)
    return np.array([df_dx, df_dy])

def hess_f_c(point):
    """
    Using d/dw[2*ri*gi] = 2*(d ri/dw)*gi + 2*ri*(d gi/dw)
    where gi = d ri/d(other variable).

    H_xx = 2*(y-1)^2 + 2*(y^2-1)^2 + 2*(y^3-1)^2
           (second partials of r1,r2,r3 w.r.t. x are all zero)

    H_yy = 2*x^2 + 2*(2xy)^2 + 4*x*r2 + 2*(3xy^2)^2 + 12*x*y*r3
           (using d^2 r1/dy^2 = 0, d^2 r2/dy^2 = 2x, d^2 r3/dy^2 = 6xy)

    H_xy = 2*(y-1)*x + 2*r1
         + 2*(y^2-1)*2xy + 4*y*r2
         + 2*(y^3-1)*3xy^2 + 6*y^2*r3
    """
    x, y = point
    r1 = 1.5   - x + x*y
    r2 = 2.25  - x + x*y**2
    r3 = 2.625 - x + x*y**3

    H_xx = 2*(y-1)**2 + 2*(y**2-1)**2 + 2*(y**3-1)**2

    H_yy = (2*x**2
            + 2*(2*x*y)**2 + 4*x*r2
            + 2*(3*x*y**2)**2 + 12*x*y*r3)

    H_xy = (2*(y-1)*x    + 2*r1
            + 2*(y**2-1)*2*x*y + 4*y*r2
            + 2*(y**3-1)*3*x*y**2 + 6*y**2*r3)

    return np.array([
        [H_xx, H_xy],
        [H_xy, H_yy],
    ], dtype=float)

STARTING_POINTS_C = [
    np.array([1.0, 1.0]),
    np.array([4.5, 4.5]),
]


# ============================================================
# PROBLEM 6: COMPARISON
# ============================================================

# --- method registry ---
# each entry: (display name, run function, needs hessian)
# Newton needs hess passed separately, all others only need grad

METHODS = [
    ("GD",       run_gd,          False),
    ("Polyak",   run_polyak_gd,   False),
    ("Nesterov", run_nesterov_gd, False),
    ("AdaGrad",  run_adagrad,     False),
    ("Adam",     run_adam,        False),
    ("Newton",   run_newton,      True),
    ("BFGS",     run_bfgs,        False),
]

# --- function registry ---
# each entry: (label, description, f, grad, hess, starting points)

FUNCTIONS = [
    (
        "1)",
        "f(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y",
        f_a, grad_f_a, hess_f_a, STARTING_POINTS_A,
    ),
    (
        "2)",
        "f(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2",
        f_b, grad_f_b, hess_f_b, STARTING_POINTS_B,
    ),
    (
        "3)",
        "f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2",
        f_c, grad_f_c, hess_f_c, STARTING_POINTS_C,
    ),
]

STEP_CHECKPOINTS = [2, 5, 10, 100]
TIME_CHECKPOINTS = [0.1, 1.0, 2.0]   # seconds


def call_method(run_fn, needs_hess, f, grad, hess, x, n_steps):
    """Dispatch a method call with or without the Hessian argument."""
    if needs_hess:
        return run_fn(x, grad, hess, f, n_steps)
    return run_fn(x, grad, f, n_steps)


def run_for_steps(run_fn, needs_hess, f, grad, hess, x1, n_steps):
    """Run a method for exactly n_steps steps. Returns final f value."""
    try:
        _, f_history = call_method(run_fn, needs_hess, f, grad, hess, x1, n_steps)
        val = f_history[-1]
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def run_for_time(run_fn, needs_hess, f, grad, hess, x1, time_budget):
    """
    Run a method in batches until time_budget seconds have elapsed.
    Uses a doubling batch strategy: start with 10 steps, double each batch
    so we do not waste time calling the function one step at a time.
    Returns the best (lowest finite) f value seen.
    """
    best_f   = np.inf
    n_batch  = 10
    x        = x1.copy()
    deadline = time.perf_counter() + time_budget

    while time.perf_counter() < deadline:
        try:
            x_next, f_history = call_method(run_fn, needs_hess, f, grad, hess, x, n_batch)

            for val in f_history:
                if np.isfinite(val) and val < best_f:
                    best_f = val

            # only continue from x_next if it gave a finite value
            if np.isfinite(f_history[-1]):
                x = x_next
            else:
                break

        except Exception:
            break

        n_batch = min(n_batch * 2, 1000)   # double batch size up to 1000

    return best_f


def format_f(val):
    """Format a function value for printing, handling inf and nan."""
    if not np.isfinite(val):
        return "diverged"
    if abs(val) < 1e-10:
        return f"{val:.2e}"
    return f"{val:.6f}"


def print_ranking(results_row, label):
    """Print top 3 methods by lowest f value."""
    finite   = [(name, v) for name, v in results_row if np.isfinite(v)]
    infinite = [(name, v) for name, v in results_row if not np.isfinite(v)]
    ranked   = sorted(finite, key=lambda t: t[1]) + infinite

    top3 = ranked[:3]
    parts = [f"{i+1}. {name} ({format_f(v)})" for i, (name, v) in enumerate(top3)]
    print(f"    {label}: {',  '.join(parts)}")


def run_comparison():

    # --------------------------------------------------------
    # PART (a): best method at 2, 5, 10, 100 steps
    # --------------------------------------------------------

    print("=" * 70)
    print("PART (a): Performance by number of steps")
    print("=" * 70)

    for fn_label, fn_desc, f, grad, hess, starting_points in FUNCTIONS:

        print(f"\nFunction {fn_label}  {fn_desc}")
        print("-" * 70)

        for x1 in starting_points:
            print(f"\n  Starting point: {x1}")
            print(f"  {'Method':<12}", end="")
            for k in STEP_CHECKPOINTS:
                print(f"  {k:>8} steps", end="")
            print()
            print(f"  {'':-<12}", end="")
            for _ in STEP_CHECKPOINTS:
                print(f"  {'':->13}", end="")
            print()

            step_results = {name: {} for name, _, _ in METHODS}
            for method_name, run_fn, needs_hess in METHODS:
                for n_steps in STEP_CHECKPOINTS:
                    step_results[method_name][n_steps] = run_for_steps(
                        run_fn, needs_hess,
                        f, grad, hess, x1, n_steps
                    )
                print(f"  {method_name:<12}", end="")
                for n_steps in STEP_CHECKPOINTS:
                    print(f"  {format_f(step_results[method_name][n_steps]):>13}", end="")
                print()

            print()
            for n_steps in STEP_CHECKPOINTS:
                row = [(name, step_results[name][n_steps]) for name, _, _ in METHODS]
                print_ranking(row, f"{n_steps:>3} steps")

    # --------------------------------------------------------
    # PART (b): best method after 0.1, 1, 2 seconds
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PART (b): Performance by wall-clock time")
    print("=" * 70)

    for fn_label, fn_desc, f, grad, hess, starting_points in FUNCTIONS:

        print(f"\nFunction {fn_label}  {fn_desc}")
        print("-" * 70)

        for x1 in starting_points:
            print(f"\n  Starting point: {x1}")
            print(f"  {'Method':<12}", end="")
            for t in TIME_CHECKPOINTS:
                print(f"  {t:>7.1f}s", end="")
            print()
            print(f"  {'':-<12}", end="")
            for _ in TIME_CHECKPOINTS:
                print(f"  {'':->9}", end="")
            print()

            time_results = {name: {} for name, _, _ in METHODS}
            for method_name, run_fn, needs_hess in METHODS:
                for t_budget in TIME_CHECKPOINTS:
                    time_results[method_name][t_budget] = run_for_time(
                        run_fn, needs_hess,
                        f, grad, hess, x1, t_budget
                    )
                print(f"  {method_name:<12}", end="")
                for t_budget in TIME_CHECKPOINTS:
                    print(f"  {format_f(time_results[method_name][t_budget]):>9}", end="")
                print()

            print()
            for t_budget in TIME_CHECKPOINTS:
                row = [(name, time_results[name][t_budget]) for name, _, _ in METHODS]
                print_ranking(row, f"{t_budget:.1f}s")


# ============================================================
# PROBLEM 7: LINEAR REGRESSION
# ============================================================

def run_sgd(x1, grad_single, f, n_steps, N, gamma=0.001):
    """
    SGD: at each step use the gradient of one randomly chosen term f_j.

        j ~ Uniform{0, ..., N-1}
        x_{k+1} = x_k - gamma * grad_f_j(x_k)

    Cheap per step (O(1) gradient evaluations) vs GD's O(N), useful for large N.

    Parameters
    ----------
    x1          : starting point
    grad_single : function(x, j) returning gradient of the j-th term at x
    f           : full objective, used only to record history
    n_steps     : number of steps
    N           : number of terms in the sum
    gamma       : learning rate (default 0.001)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    f_history = [f(x)]
    rng       = np.random.default_rng(seed=0)   # fixed seed for reproducibility

    for _ in range(n_steps):
        j = rng.integers(0, N)
        x = x - gamma * grad_single(x, j)
        f_history.append(f(x))

    return x, f_history


def run_lbfgs(x1, grad, f, n_steps, m=10, eps=1e-10, c=1e-4, tau=0.5):
    """
    L-BFGS: limited memory BFGS via the two-loop recursion (Algorithm 1, lecture notes).

    Stores only the last m pairs (delta_k, y_k) and reconstructs B_k * grad_f on
    the fly, reducing memory from O(n^2) to O(mn). Includes backtracking line search
    as required for all quasi-Newton methods.

    Parameters
    ----------
    x1      : starting point
    grad    : gradient function
    f       : objective function
    n_steps : number of steps
    m       : number of stored curvature pairs (default 10)
    eps     : minimum delta^T y to apply update (default 1e-10)
    c       : Armijo sufficient decrease constant (default 1e-4)
    tau     : backtracking step reduction factor (default 0.5)

    Returns
    -------
    x         : final iterate
    f_history : list of f(x_k) values, length n_steps + 1
    """
    x         = x1.copy()
    f_history = [f(x)]
    deltas    = []   # delta_k = x_k - x_{k-1}
    ys        = []   # y_k     = grad_f(x_k) - grad_f(x_{k-1})
    rhos      = []   # rho_k   = 1 / (delta_k^T y_k)
    grad_x    = grad(x)

    for _ in range(n_steps):

        # two-loop recursion: compute B_k * grad_x
        q      = grad_x.copy()
        alphas = []
        for delta, y, rho in zip(reversed(deltas), reversed(ys), reversed(rhos)):
            alpha = rho * (delta @ q)
            q     = q - alpha * y
            alphas.append(alpha)

        scale = (deltas[-1] @ ys[-1]) / (ys[-1] @ ys[-1]) if deltas else 1.0
        r     = scale * q

        for delta, y, rho, alpha in zip(deltas, ys, rhos, reversed(alphas)):
            beta = rho * (y @ r)
            r    = r + delta * (alpha - beta)

        direction = r

        # backtracking line search
        alpha_ls = 1.0
        f_x      = f(x)
        slope    = grad_x @ direction
        while True:
            f_candidate = f(x - alpha_ls * direction)
            if np.isfinite(f_candidate) and f_candidate <= f_x - c * alpha_ls * slope:
                break
            alpha_ls *= tau
            if alpha_ls < 1e-16:
                break

        x_next    = x - alpha_ls * direction
        grad_next = grad(x_next)

        delta       = x_next - x
        y           = grad_next - grad_x
        delta_dot_y = delta @ y
        if delta_dot_y > eps:
            deltas.append(delta)
            ys.append(y)
            rhos.append(1.0 / delta_dot_y)
            if len(deltas) > m:
                deltas.pop(0)
                ys.pop(0)
                rhos.pop(0)

        x      = x_next
        grad_x = grad_next
        f_history.append(f(x))

    return x, f_history


def make_regression(N, seed=42):
    """
    Generate N points (i, i + nu_i) where nu_i ~ Uniform[0, 1].
    Returns the MSE loss, gradient, Hessian, single-point gradient for SGD
    and the closed-form optimal parameters (k*, n*).

    The model is g(x) = k*i + n with parameters theta = [k, n].
    The Hessian is constant (loss is quadratic) and precomputed once.
    """
    rng = np.random.default_rng(seed=seed)
    xs  = np.arange(1, N + 1, dtype=float)
    ys  = xs + rng.uniform(0.0, 1.0, size=N)

    H = np.array([
        [2 * np.mean(xs**2), 2 * np.mean(xs)],
        [2 * np.mean(xs),    2.0            ],
    ])

    A          = np.column_stack([xs, np.ones(N)])
    theta_star = np.linalg.lstsq(A, ys, rcond=None)[0]

    def f(theta):
        k, n = theta
        return float(np.mean((k * xs + n - ys)**2))

    def grad_f(theta):
        k, n = theta
        r    = k * xs + n - ys
        return np.array([2 * np.mean(r * xs), 2 * np.mean(r)])

    def hess_f(theta):
        return H

    def grad_single(theta, j):
        k, n = theta
        r_j  = k * xs[j] + n - ys[j]
        return np.array([2 * r_j * xs[j], 2 * r_j])

    return f, grad_f, hess_f, grad_single, theta_star


# --- method registry for problem 7 ---
# SGD has a different signature (needs grad_single and N) so it is handled separately

REGRESSION_N_VALUES = [50, 100, 1_000, 10_000, 100_000, 1_000_000]
REGRESSION_METHODS  = ["GD", "SGD", "Newton", "BFGS", "L-BFGS"]
REGRESSION_STEPS    = [2, 5, 10, 100]


def call_regression_method(method, f, grad_f, hess_f, grad_single, N, x, n_steps):
    """Dispatch a single regression method call for n_steps steps."""
    beta = np.linalg.eigvalsh(hess_f(x)).max()

    gamma_gd = 1.0 / beta
    gamma_sgd = 0.01 / beta

    if method == "GD":
        return run_gd(x, grad_f, f, n_steps, gamma=gamma_gd)
    if method == "SGD":
        return run_sgd(x, grad_single, f, n_steps, N, gamma=gamma_sgd)
    if method == "Newton":
        return run_newton(x, grad_f, hess_f, f, n_steps)
    if method == "BFGS":
        return run_bfgs(x, grad_f, f, n_steps)
    if method == "L-BFGS":
        return run_lbfgs(x, grad_f, f, n_steps)
    raise ValueError(f"Unknown method: {method}")


def run_reg_for_steps(method, f, grad_f, hess_f, grad_single, N, x1, n_steps):
    """Run a regression method for n_steps steps. Returns final loss value."""
    try:
        _, h = call_regression_method(method, f, grad_f, hess_f, grad_single, N, x1, n_steps)
        val  = h[-1]
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def run_regression_comparison():

    x1 = np.array([0.0, 0.0])   # start at k=0, n=0 for all methods and all N

    print("=" * 75)
    print("PROBLEM 7: Linear Regression  g(x) = kx + n")
    print("=" * 75)

    for N in REGRESSION_N_VALUES:
        f, grad_f, hess_f, grad_single, theta_star = make_regression(N)
        f_star = f(theta_star)

        print(f"\nN = {N:>9}   optimum = {f_star:.6f}"
              f"   k* = {theta_star[0]:.4f}   n* = {theta_star[1]:.4f}")
        print("-" * 75)
        print(f"  {'Method':<10}", end="")
        for k in REGRESSION_STEPS:
            print(f"  {k:>8} steps", end="")
        print()
        print(f"  {'':-<10}", end="")
        for _ in REGRESSION_STEPS:
            print(f"  {'':->13}", end="")
        print()

        step_results = {m: {} for m in REGRESSION_METHODS}
        for method in REGRESSION_METHODS:
            for n_steps in REGRESSION_STEPS:
                step_results[method][n_steps] = run_reg_for_steps(
                    method, f, grad_f, hess_f, grad_single, N, x1, n_steps
                )
            print(f"  {method:<10}", end="")
            for n_steps in REGRESSION_STEPS:
                print(f"  {format_f(step_results[method][n_steps]):>13}", end="")
            print()

        print()
        for n_steps in REGRESSION_STEPS:
            row = [(m, step_results[m][n_steps]) for m in REGRESSION_METHODS]
            print_ranking(row, f"{n_steps:>3} steps")


if __name__ == "__main__":
    run_comparison()           # Problem 6: compare methods on three functions
    print()
    run_regression_comparison()  # Problem 7: linear regression across N values
