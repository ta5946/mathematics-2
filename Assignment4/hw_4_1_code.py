"""Optimization methods for MAT2 Homework 4.1.

The functions in this file are intentionally dimension-independent whenever the
method allows it. In particular, Nelder-Mead infers the dimension from x0 and
constructs an n-dimensional simplex automatically.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

TABLE_WIDTH = 60


# ============================================================
# Helpers
# ============================================================

def _as_vector(x):
    """Convert input to a one-dimensional floating point NumPy array."""
    return np.asarray(x, dtype=float).reshape(-1)


class CallCounter:
    """Wrap a function and count how many times it is called."""

    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.fn(x)

    def reset(self):
        self.count = 0


# ============================================================
# Nelder-Mead method
# ============================================================

def nelder_mead(
    f,
    x0,
    diameter=1.0,
    max_iter=1000,
    tol_x=1e-8,
    tol_f=1e-10,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
):
    """Minimize f using the Nelder-Mead simplex method.

    Parameters
    ----------
    f : callable
        Function accepting a vector and returning a scalar.
    x0 : array-like
        One point of the initial simplex.
    diameter : float
        Distance from x0 to each coordinate-shifted simplex point.
    max_iter : int
        Maximum number of simplex iterations.
    tol_x, tol_f : float
        Stop when both simplex diameter and function-value spread are small.
    alpha, gamma, rho, sigma : float
        Reflection, expansion, contraction, and shrink coefficients.
    """
    x0 = _as_vector(x0)
    n = len(x0)

    simplex = np.vstack([x0, x0 + diameter * np.eye(n)])
    values = np.array([f(x) for x in simplex], dtype=float)
    function_evaluations = n + 1
    history = []

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        best = simplex[0].copy()
        worst = simplex[-1].copy()
        best_value = values[0]
        worst_value = values[-1]
        second_worst_value = values[-2]

        simplex_diameter = max(np.linalg.norm(x - best) for x in simplex)
        value_spread = worst_value - best_value
        history.append(
            {
                "x": best.copy(),
                "f": float(best_value),
                "simplex_diameter": float(simplex_diameter),
                "value_spread": float(value_spread),
            }
        )

        if simplex_diameter < tol_x and value_spread < tol_f:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        reflected = centroid + alpha * (centroid - worst)
        reflected_value = f(reflected)
        function_evaluations += 1

        if values[0] <= reflected_value < second_worst_value:
            simplex[-1] = reflected
            values[-1] = reflected_value
            continue

        if reflected_value < values[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded_value = f(expanded)
            function_evaluations += 1

            if expanded_value < reflected_value:
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
            continue

        if reflected_value < worst_value:
            # Outside contraction
            contracted = centroid + rho * (reflected - centroid)
        else:
            # Inside contraction
            contracted = centroid + rho * (worst - centroid)

        contracted_value = f(contracted)
        function_evaluations += 1

        if contracted_value < min(reflected_value, worst_value):
            simplex[-1] = contracted
            values[-1] = contracted_value
            continue

        # Shrink all points except the best point toward the best point.
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            values[i] = f(simplex[i])
        function_evaluations += n

    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]

    return {
        "method": "Nelder-Mead",
        "x": simplex[0].copy(),
        "f": float(values[0]),
        "iterations": len(history),
        "function_evaluations": function_evaluations,
        "history": history,
        "simplex": simplex.copy(),
        "simplex_values": values.copy(),
    }


# ============================================================
# Gradient-based methods from Homework GD.2
# ============================================================

def gradient_descent(f, grad, x0, lr=1e-3, max_iter=1000, tol_grad=1e-8):
    """Standard gradient descent with fixed learning rate."""
    x = _as_vector(x0)
    history = []
    iterations = 0

    for _ in range(max_iter):
        value = float(f(x))
        g = _as_vector(grad(x))
        history.append({"x": x.copy(), "f": value, "grad_norm": float(np.linalg.norm(g))})

        if np.linalg.norm(g) < tol_grad:
            break

        x = x - lr * g
        iterations += 1

    return {
        "method": "Gradient descent",
        "x": x.copy(),
        "f": float(f(x)),
        "iterations": iterations,
        "function_evaluations": len(history) + 1,
        "gradient_evaluations": len(history),
        "history": history,
    }


def polyak_gd(
    f,
    grad,
    x0,
    lr=1e-3,
    momentum=0.9,
    max_iter=1000,
    tol_grad=1e-8,
):
    """Heavy-ball gradient descent with Polyak momentum."""
    x = _as_vector(x0)
    velocity = np.zeros_like(x)
    history = []
    iterations = 0

    for _ in range(max_iter):
        value = float(f(x))
        g = _as_vector(grad(x))
        history.append({"x": x.copy(), "f": value, "grad_norm": float(np.linalg.norm(g))})

        if np.linalg.norm(g) < tol_grad:
            break

        velocity = momentum * velocity - lr * g
        x = x + velocity
        iterations += 1

    return {
        "method": "Polyak GD",
        "x": x.copy(),
        "f": float(f(x)),
        "iterations": iterations,
        "function_evaluations": len(history) + 1,
        "gradient_evaluations": len(history),
        "history": history,
    }


def nesterov_gd(
    f,
    grad,
    x0,
    lr=1e-3,
    momentum=0.9,
    max_iter=1000,
    tol_grad=1e-8,
):
    """Nesterov accelerated gradient descent."""
    x = _as_vector(x0)
    velocity = np.zeros_like(x)
    history = []
    iterations = 0

    for _ in range(max_iter):
        lookahead = x + momentum * velocity
        g = _as_vector(grad(lookahead))
        value = float(f(x))
        history.append({"x": x.copy(), "f": value, "grad_norm": float(np.linalg.norm(g))})

        if np.linalg.norm(g) < tol_grad:
            break

        velocity = momentum * velocity - lr * g
        x = x + velocity
        iterations += 1

    return {
        "method": "Nesterov GD",
        "x": x.copy(),
        "f": float(f(x)),
        "iterations": iterations,
        "function_evaluations": len(history) + 1,
        "gradient_evaluations": len(history),
        "history": history,
    }


def adagrad(
    f,
    grad,
    x0,
    lr=1e-2,
    epsilon=1e-8,
    max_iter=1000,
    tol_grad=1e-8,
):
    """AdaGrad with coordinate-wise accumulated squared gradients."""
    x = _as_vector(x0)
    accumulated = np.zeros_like(x)
    history = []
    iterations = 0

    for _ in range(max_iter):
        value = float(f(x))
        g = _as_vector(grad(x))
        history.append({"x": x.copy(), "f": value, "grad_norm": float(np.linalg.norm(g))})

        if np.linalg.norm(g) < tol_grad:
            break

        accumulated += g * g
        x = x - lr * g / (np.sqrt(accumulated) + epsilon)
        iterations += 1

    return {
        "method": "AdaGrad",
        "x": x.copy(),
        "f": float(f(x)),
        "iterations": iterations,
        "function_evaluations": len(history) + 1,
        "gradient_evaluations": len(history),
        "history": history,
    }


def newton(
    f,
    grad,
    hess,
    x0,
    step_scale=1.0,
    regularization=0.0,
    max_iter=100,
    tol_grad=1e-8,
):
    """Newton's method using a linear solve for the Newton direction."""
    x = _as_vector(x0)
    history = []
    iterations = 0

    for _ in range(max_iter):
        value = float(f(x))
        g = _as_vector(grad(x))
        grad_norm = np.linalg.norm(g)
        history.append({"x": x.copy(), "f": value, "grad_norm": float(grad_norm)})

        if grad_norm < tol_grad:
            break

        h = np.asarray(hess(x), dtype=float)
        if regularization:
            h = h + regularization * np.eye(len(x))

        try:
            direction = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(h, g, rcond=None)[0]

        x = x - step_scale * direction
        iterations += 1

    return {
        "method": "Newton",
        "x": x.copy(),
        "f": float(f(x)),
        "iterations": iterations,
        "function_evaluations": len(history) + 1,
        "gradient_evaluations": len(history),
        "hessian_evaluations": len(history),
        "history": history,
    }


def bfgs(
    f,
    grad,
    x0,
    max_iter=1000,
    tol_grad=1e-8,
    initial_inverse_hessian=None,
    armijo_c=1e-4,
    backtrack=0.5,
    min_step=1e-12,
    curvature_min=1e-10,
):
    """BFGS quasi-Newton method with Armijo backtracking line search."""
    x = _as_vector(x0)
    n = len(x)
    inverse_hessian = (
        np.eye(n)
        if initial_inverse_hessian is None
        else np.asarray(initial_inverse_hessian, dtype=float)
    )
    history = []
    function_evaluations = 0
    gradient_evaluations = 0
    iterations = 0

    value = float(f(x))
    function_evaluations += 1
    g = _as_vector(grad(x))
    gradient_evaluations += 1

    for _ in range(max_iter):
        grad_norm = np.linalg.norm(g)
        history.append({"x": x.copy(), "f": value, "grad_norm": float(grad_norm)})

        if grad_norm < tol_grad:
            break

        direction = -inverse_hessian @ g
        if np.dot(direction, g) >= 0:
            direction = -g
            inverse_hessian = np.eye(n)

        step = 1.0
        directional_derivative = np.dot(g, direction)
        line_search_failed = False

        while step >= min_step:
            candidate = x + step * direction
            candidate_value = float(f(candidate))
            function_evaluations += 1

            if candidate_value <= value + armijo_c * step * directional_derivative:
                break

            step *= backtrack
        else:
            candidate = x
            candidate_value = value
            line_search_failed = True

        candidate_grad = _as_vector(grad(candidate))
        gradient_evaluations += 1

        if line_search_failed:
            break

        s = candidate - x
        y = candidate_grad - g
        curvature = np.dot(y, s)

        if curvature > curvature_min:
            rho = 1.0 / curvature
            identity = np.eye(n)
            sy = np.outer(s, y)
            ys = np.outer(y, s)
            ss = np.outer(s, s)
            inverse_hessian = (
                (identity - rho * sy)
                @ inverse_hessian
                @ (identity - rho * ys)
                + rho * ss
            )
        else:
            inverse_hessian = np.eye(n)

        x = candidate
        value = candidate_value
        g = candidate_grad
        iterations += 1

    return {
        "method": "BFGS",
        "x": x.copy(),
        "f": float(value),
        "iterations": iterations,
        "function_evaluations": function_evaluations,
        "gradient_evaluations": gradient_evaluations,
        "history": history,
    }


# ============================================================
# Test functions from Homework GD.2, Problem 6
# ============================================================

def f_a(point):
    """f(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y."""
    x, y, z = _as_vector(point)
    return (x - z) ** 2 + (2 * y + z) ** 2 + (4 * x - 2 * y + z) ** 2 + x + y


def grad_a(point):
    """Gradient of f_a."""
    x, y, z = _as_vector(point)
    r1 = x - z
    r2 = 2 * y + z
    r3 = 4 * x - 2 * y + z

    return np.array([
        2 * r1 + 8 * r3 + 1,
        4 * r2 - 4 * r3 + 1,
        -2 * r1 + 2 * r2 + 2 * r3,
    ], dtype=float)


def hess_a(point):
    """Hessian of f_a. It is constant because f_a is quadratic."""
    return np.array([
        [34, -16, 6],
        [-16, 16, 0],
        [6, 0, 6],
    ], dtype=float)


def f_b(point):
    """f(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2."""
    x, y, z = _as_vector(point)
    return (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x**2) ** 2 + 100 * (z - y**2) ** 2


def grad_b(point):
    """Gradient of f_b."""
    x, y, z = _as_vector(point)
    u = y - x**2
    v = z - y**2

    return np.array([
        2 * (x - 1) - 400 * x * u,
        2 * (y - 1) + 200 * u - 400 * y * v,
        200 * v,
    ], dtype=float)


def hess_b(point):
    """Hessian of f_b."""
    x, y, z = _as_vector(point)
    u = y - x**2
    v = z - y**2

    return np.array([
        [2 - 400 * u + 800 * x**2, -400 * x, 0],
        [-400 * x, 202 - 400 * v + 800 * y**2, -400 * y],
        [0, -400 * y, 200],
    ], dtype=float)


def f_c(point):
    """Beale function from GD.2."""
    x, y = _as_vector(point)
    r1 = 1.5 - x + x * y
    r2 = 2.25 - x + x * y**2
    r3 = 2.625 - x + x * y**3
    return r1**2 + r2**2 + r3**2


def grad_c(point):
    """Gradient of f_c."""
    x, y = _as_vector(point)
    r1 = 1.5 - x + x * y
    r2 = 2.25 - x + x * y**2
    r3 = 2.625 - x + x * y**3

    return np.array([
        2 * r1 * (y - 1) + 2 * r2 * (y**2 - 1) + 2 * r3 * (y**3 - 1),
        2 * r1 * x + 2 * r2 * (2 * x * y) + 2 * r3 * (3 * x * y**2),
    ], dtype=float)


def hess_c(point):
    """Hessian of f_c."""
    x, y = _as_vector(point)
    r1 = 1.5 - x + x * y
    r2 = 2.25 - x + x * y**2
    r3 = 2.625 - x + x * y**3

    h_xx = 2 * (y - 1) ** 2 + 2 * (y**2 - 1) ** 2 + 2 * (y**3 - 1) ** 2
    h_yy = (
        2 * x**2
        + 2 * (2 * x * y) ** 2
        + 4 * x * r2
        + 2 * (3 * x * y**2) ** 2
        + 12 * x * y * r3
    )
    h_xy = (
        2 * (y - 1) * x
        + 2 * r1
        + 2 * (y**2 - 1) * (2 * x * y)
        + 4 * y * r2
        + 2 * (y**3 - 1) * (3 * x * y**2)
        + 6 * y**2 * r3
    )

    return np.array([
        [h_xx, h_xy],
        [h_xy, h_yy],
    ], dtype=float)


STARTING_POINTS_A = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
]

STARTING_POINTS_B = [
    np.array([1.2, 1.2, 1.2]),
    np.array([-1.0, 1.2, 1.2]),
]

STARTING_POINTS_C = [
    np.array([1.0, 1.0]),
    np.array([4.5, 4.5]),
]

TEST_FUNCTIONS = [
    ("f_a", f_a, grad_a, hess_a, STARTING_POINTS_A),
    ("f_b", f_b, grad_b, hess_b, STARTING_POINTS_B),
    ("f_c", f_c, grad_c, hess_c, STARTING_POINTS_C),
]


# Method registry used later for comparison tables.
# The last flag says whether the method needs Hessian information.
GRADIENT_METHODS = [
    ("GD", gradient_descent, False),
    ("Polyak GD", polyak_gd, False),
    ("Nesterov GD", nesterov_gd, False),
    ("AdaGrad", adagrad, False),
    ("Newton", newton, True),
    ("BFGS", bfgs, False),
]

NELDER_MEAD_DIAMETERS = [0.1, 1.0, 3.0]


# ============================================================
# Comparison printing
# ============================================================

def _format_float(value, width=12):
    """Format numbers compactly for terminal tables."""
    if not np.isfinite(value):
        return f"{'diverged':>{width}}"
    if abs(value) >= 1e5 or (0 < abs(value) < 1e-4):
        return f"{value:>{width}.3e}"
    return f"{value:>{width}.6f}"


def _format_vector(x, precision=6):
    """Format a vector on one line."""
    return "[" + ", ".join(f"{v:.{precision}g}" for v in _as_vector(x)) + "]"


def _known_minimum_value(f, grad, hess, starting_points):
    """Return a reference minimum value used only for reporting quality."""
    n = len(starting_points[0])

    if f is f_a:
        x_star = -np.linalg.solve(hess(np.zeros(n)), grad(np.zeros(n)))
        return float(f(x_star))

    # Both f_b and f_c have known global minimum value 0.
    return 0.0


def _run_gradient_method(name, method, needs_hessian, f, grad, hess, x0, max_iter):
    """Run one GD.2 method and normalize its result dictionary."""
    start = time.perf_counter()
    if needs_hessian:
        result = method(f, grad, hess, x0, max_iter=max_iter)
    else:
        result = method(f, grad, x0, max_iter=max_iter)
    elapsed = time.perf_counter() - start

    total_evaluations = (
        result.get("function_evaluations", 0)
        + result.get("gradient_evaluations", 0)
        + result.get("hessian_evaluations", 0)
    )

    return {
        "method": name,
        "x": result["x"],
        "f": result["f"],
        "iterations": result["iterations"],
        "function_evaluations": result.get("function_evaluations", 0),
        "gradient_evaluations": result.get("gradient_evaluations", 0),
        "hessian_evaluations": result.get("hessian_evaluations", 0),
        "total_evaluations": total_evaluations,
        "elapsed": elapsed,
        "evaluations_per_second": total_evaluations / elapsed if elapsed > 0 else np.inf,
    }


def _run_nelder_mead(f, x0, diameter, max_iter):
    """Run Nelder-Mead and normalize its result dictionary."""
    start = time.perf_counter()
    result = nelder_mead(f, x0, diameter=diameter, max_iter=max_iter)
    elapsed = time.perf_counter() - start

    return {
        "method": f"Nelder-Mead d={diameter:g}",
        "x": result["x"],
        "f": result["f"],
        "iterations": result["iterations"],
        "function_evaluations": result["function_evaluations"],
        "gradient_evaluations": 0,
        "hessian_evaluations": 0,
        "total_evaluations": result["function_evaluations"],
        "elapsed": elapsed,
        "evaluations_per_second": (
            result["function_evaluations"] / elapsed if elapsed > 0 else np.inf
        ),
    }


def _print_parameter_summary():
    """Print the parameters used in the comparison."""
    print("Parameters")
    print("-" * TABLE_WIDTH)
    print("GD: lr=0.001")
    print("Polyak GD: lr=0.001, momentum=0.9")
    print("Nesterov GD: lr=0.001, momentum=0.9")
    print("AdaGrad: lr=0.01, epsilon=1e-8")
    print("Newton: step_scale=1.0, regularization=0.0")
    print("BFGS: Armijo line search, c=1e-4, backtrack=0.5")
    print(f"Nelder-Mead: diameters={NELDER_MEAD_DIAMETERS}, alpha=1, gamma=2, rho=0.5, sigma=0.5")
    print()


def _print_information_summary():
    """Print what information each method uses."""
    print("Information used")
    print("-" * TABLE_WIDTH)
    print("Nelder-Mead: function values only")
    print("GD / Polyak GD / Nesterov GD / AdaGrad / BFGS: function values and gradients")
    print("Newton: function values, gradients, and Hessians")
    print()


def compare_methods(max_iter=1000):
    """Print compact comparison tables for all methods and all test functions."""
    old_numpy_settings = np.seterr(all="ignore")

    try:
        _print_information_summary()
        _print_parameter_summary()

        summary = []

        for function_name, f, grad, hess, starting_points in TEST_FUNCTIONS:
            f_star = _known_minimum_value(f, grad, hess, starting_points)

            print("=" * TABLE_WIDTH)
            print(f"{function_name}: reference minimum value approximately {f_star:.12g}")
            print("=" * TABLE_WIDTH)

            for start_index, x0 in enumerate(starting_points, start=1):
                rows = []

                for method_name, method, needs_hessian in GRADIENT_METHODS:
                    rows.append(
                        _run_gradient_method(
                            method_name, method, needs_hessian, f, grad, hess, x0, max_iter
                        )
                    )

                for diameter in NELDER_MEAD_DIAMETERS:
                    rows.append(_run_nelder_mead(f, x0, diameter, max_iter))

                rows.sort(key=lambda row: row["f"] if np.isfinite(row["f"]) else np.inf)

                print(f"\nStarting point {start_index}: x0 = {_format_vector(x0)}")
                print(
                    f"{'method':<20} {'f(x)':>12} {'gap':>12} {'iter':>6} "
                    f"{'evals':>7} {'evals/s':>10} {'final x'}"
                )
                print("-" * TABLE_WIDTH)

                for row in rows:
                    gap = row["f"] - f_star
                    print(
                        f"{row['method']:<20} "
                        f"{_format_float(row['f'])} "
                        f"{_format_float(gap)} "
                        f"{row['iterations']:>6} "
                        f"{row['total_evaluations']:>7} "
                        f"{row['evaluations_per_second']:>10.0f} "
                        f"{_format_vector(row['x'])}"
                    )

                best = rows[0]
                summary.append((function_name, start_index, best["method"], best["f"]))
                print(f"\nBest final value: {best['method']} with f = {best['f']:.12g}")

        print("\n" + "=" * TABLE_WIDTH)
        print("Generality across functions")
        print("=" * TABLE_WIDTH)
        for function_name, start_index, method_name, value in summary:
            print(
                f"{function_name}, starting point {start_index}: "
                f"{method_name} gave the best final value ({value:.12g})"
            )
    finally:
        np.seterr(**old_numpy_settings)


# ============================================================
# Black-box optimization
# ============================================================

STUDENT_ID = "63210005"
BLACK_BOX_EXE = ".\\hw_4_1_win.exe"
BLACK_BOX_STARTS = [
    np.array([0.0, 0.0, 0.0]),
]


def black_box_value(function_index, point, student_id=STUDENT_ID, executable=BLACK_BOX_EXE):
    """Evaluate the provided black-box executable at one point."""
    x, y, z = _as_vector(point)
    command = [
        executable,
        str(student_id),
        str(function_index),
        f"{x:.17g}",
        f"{y:.17g}",
        f"{z:.17g}",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def central_difference_gradient(f, x, h=1e-5, workers=6):
    """Approximate the gradient by two-sided finite differences."""
    x = _as_vector(x)
    points = []

    for j in range(len(x)):
        step = np.zeros_like(x)
        step[j] = h
        points.append(x + step)
        points.append(x - step)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(f, points))

    grad = np.zeros_like(x)
    for j in range(len(x)):
        grad[j] = (values[2 * j] - values[2 * j + 1]) / (2 * h)

    return grad


def make_black_box_objective(function_index):
    """Create a cached callable for one black-box function."""
    cache = {}
    stats = {"calls": 0}

    def f(x):
        key = tuple(round(float(v), 12) for v in _as_vector(x))
        if key not in cache:
            cache[key] = black_box_value(function_index, key)
            stats["calls"] += 1
        return cache[key]

    f.stats = stats
    f.cache = cache
    return f


def black_box_bfgs(
    function_index,
    x0,
    max_iter=20,
    grad_h=1e-5,
    tol_grad=1e-7,
    min_step=1e-7,
):
    """BFGS for a black-box function using central finite-difference gradients."""
    f = make_black_box_objective(function_index)

    def grad(x):
        return central_difference_gradient(
            f,
            x,
            h=grad_h,
            workers=2 * len(_as_vector(x)),
        )

    start = time.perf_counter()
    result = bfgs(
        f,
        grad,
        x0,
        max_iter=max_iter,
        tol_grad=tol_grad,
        min_step=min_step,
    )
    result["elapsed"] = time.perf_counter() - start
    result["black_box_calls"] = f.stats["calls"]
    result["function_index"] = function_index
    result["optimizer"] = "BFGS with finite-difference gradient"
    return result


def black_box_nelder_mead(
    function_index,
    x0,
    max_iter=20,
    diameter=1.0,
    tol_x=1e-7,
    tol_f=1e-9,
):
    """Nelder-Mead for the black-box functions."""
    f = make_black_box_objective(function_index)

    start = time.perf_counter()
    result = nelder_mead(
        f,
        x0,
        diameter=diameter,
        max_iter=max_iter,
        tol_x=tol_x,
        tol_f=tol_f,
    )
    result["elapsed"] = time.perf_counter() - start
    result["black_box_calls"] = f.stats["calls"]
    result["function_index"] = function_index
    result["diameter"] = diameter
    result["optimizer"] = f"Nelder-Mead d={diameter:g}"
    return result


def run_black_box_optimization(
    methods=None,
    function_indices=(1, 2, 3),
    starts=None,
    max_iter=20,
):
    """Run selected black-box optimization methods.

    The default runs only finite-difference BFGS from the origin because each
    executable call is slow. Extra starts can be passed explicitly when needed.
    Pass methods=(black_box_bfgs, black_box_nelder_mead) to compare both.
    """
    if methods is None:
        methods = (black_box_bfgs,)

    if starts is None:
        starts = BLACK_BOX_STARTS

    all_best = []

    for function_index in function_indices:
        print("=" * TABLE_WIDTH)
        print(f"Black-box function f_{STUDENT_ID},{function_index}")
        print("=" * TABLE_WIDTH)

        results = []

        for method in methods:
            for x0 in starts:
                if method is black_box_nelder_mead:
                    result = method(function_index, x0, max_iter=max_iter, diameter=1.0)
                else:
                    result = method(function_index, x0, max_iter=max_iter)
                results.append(result)
                print(
                    f"{result['optimizer']:<38} start={_format_vector(x0)} "
                    f"f={result['f']:.15g}, x={_format_vector(result['x'], 15)}, "
                    f"calls={result['black_box_calls']}, time={result['elapsed']:.1f}s"
                )

        best = min(results, key=lambda result: result["f"])
        all_best.append(best)
        print("\nBest candidate:")
        print(
            f"{best['optimizer']}, f={best['f']:.15g}, x={_format_vector(best['x'], 15)}, "
            f"calls={best['black_box_calls']}, time={best['elapsed']:.1f}s"
        )
        print()

    return all_best


# ============================================================
# Toy LP by active-constraint enumeration
# ============================================================

def solve_toy_lp():
    """Solve the LP by checking all triples of active constraints."""
    objective = np.array([1.0, 3.0, 4.0])
    constraints = [
        ("x1 - x2 + x3 <= 5", np.array([1.0, -1.0, 1.0]), 5.0),
        ("x1 + x2 - x3 <= 7", np.array([1.0, 1.0, -1.0]), 7.0),
        ("-x1 + x2 + x3 <= 10", np.array([-1.0, 1.0, 1.0]), 10.0),
        ("x1 + 2x2 + 3x3 <= 8", np.array([1.0, 2.0, 3.0]), 8.0),
        ("3x1 + 2x2 + x3 <= 6", np.array([3.0, 2.0, 1.0]), 6.0),
        ("x1 >= 0", np.array([-1.0, 0.0, 0.0]), 0.0),
        ("x2 >= 0", np.array([0.0, -1.0, 0.0]), 0.0),
        ("x3 >= 0", np.array([0.0, 0.0, -1.0]), 0.0),
    ]

    feasible_vertices = []

    for active_indices in combinations(range(len(constraints)), 3):
        matrix = np.vstack([constraints[i][1] for i in active_indices])
        rhs = np.array([constraints[i][2] for i in active_indices])

        try:
            x = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue

        feasible = all(a @ x <= b + 1e-10 for _, a, b in constraints)
        if feasible:
            value = float(objective @ x)
            feasible_vertices.append((value, x, active_indices))

    best_value, best_x, best_active = max(feasible_vertices, key=lambda item: item[0])

    print("=" * TABLE_WIDTH)
    print("Toy LP by active-constraint enumeration")
    print("=" * TABLE_WIDTH)
    print(f"Number of feasible vertices: {len(feasible_vertices)}")
    print(f"Optimal x = {_format_vector(best_x, 15)}")
    print(f"Optimal value = {best_value:.15g}")
    print("Active constraints:")
    for i in best_active:
        print(f"  {i + 1}. {constraints[i][0]}")

    return {
        "x": best_x,
        "value": best_value,
        "active_constraints": best_active,
        "feasible_vertices": feasible_vertices,
    }


if __name__ == "__main__":
    # compare_methods()
    # run_black_box_optimization(methods=(black_box_bfgs, black_box_nelder_mead), max_iter=20)
    solve_toy_lp()
