"""
Interior-point method for linear programming.
Mathematics 2, Part 4, Homework 2.

Problem X:
    min  -3 x1 - 4 x2
    s.t. 3 x1 + 3 x2 + 3 x3           = 4
         3 x1 +   x2        + x4      = 3
           x1 + 4 x2             + x5 = 4
         x1, ..., x5 >= 0

Functions, by homework problem:
    ip_next_step       one interior-point step                 (problem 3)
    sigma_sq           proximity-to-central-path measure
    ip_solve           iterate with fixed step until mu small   (problem 8)
    adaptive_factor    boldest safe reduction via bisection     (problem 9)
    ip_solve_adaptive  iterate with adaptive step               (problem 9)
    round_decisions    set the smaller of x_i, s_i to zero      (problem 10)
    exact_solve        exact rational solve + verification       (problem 11)
    solve_with_highs   reference solve (simplex + interior)     (problem 12)
    benchmark          iterations / runtime / error table
    plot_convergence   convergence figure

Run:  python ip_method.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Problem 3: one step of the interior-point method
# ----------------------------------------------------------------------
def ip_step(A, b, x, y, s, mu_next):
    """
    Solve system (S) for the step (h, k, f) toward target mu_next and return
    the next iterate (x', y', s').

        A h = 0
        A^T k + f = 0
        h_i s_i + f_i x_i = mu_next - x_i s_i

    Reduction (Mehlhorn-Saxena, Thm 1), with X = diag(x), S = diag(s), e = 1:
        (A S^-1 X A^T) k = b - mu_next * A S^-1 e
        f = -A^T k
        h = -X S^-1 f + mu_next S^-1 e - x
    """
    e = np.ones(len(x))
    sinv = 1.0 / s          # diagonal of S^-1
    xs = x / s              # diagonal of S^-1 X

    M = A @ (xs[:, None] * A.T)              # A S^-1 X A^T  (symmetric pos. def.)
    rhs = b - mu_next * (A @ (sinv * e))
    k = np.linalg.solve(M, rhs)
    f = -A.T @ k
    h = -xs * f + mu_next * sinv * e - x
    return x + h, y + k, s + f


def ip_next_step(A, b, c, x, y, s, mu):
    """One step with the fixed reduction mu' = (1 - 1/(6 sqrt m)) mu."""
    m = A.shape[1]
    mu_next = (1.0 - 1.0 / (6.0 * np.sqrt(m))) * mu
    x2, y2, s2 = ip_step(A, b, x, y, s, mu_next)
    return x2, y2, s2, mu_next


def sigma_sq(x, s, mu):
    """Proximity measure  sigma^2 = sum_i (x_i s_i / mu - 1)^2."""
    return float(np.sum((x * s / mu - 1.0) ** 2))


# ----------------------------------------------------------------------
# Problem 8: iterate with the fixed step until mu is small
# ----------------------------------------------------------------------
def ip_solve(A, b, c, x, y, s, mu, mu_tol=1e-9, max_iter=2000):
    """Fixed-step iteration. Returns final (x, y, s, mu) and history."""
    hist = {"it": [], "gap": [], "primal": [], "dual": []}
    for it in range(max_iter):
        hist["it"].append(it)
        hist["gap"].append(float(x @ s))
        hist["primal"].append(float(c @ x))
        hist["dual"].append(float(b @ y))
        if mu < mu_tol:
            break
        x, y, s, mu = ip_next_step(A, b, c, x, y, s, mu)
    return x, y, s, mu, hist


# ----------------------------------------------------------------------
# Problem 9: adaptive step size
# ----------------------------------------------------------------------
def adaptive_factor(A, b, x, y, s, mu, bound=1/36, tries=20):
    """
    Bisection for the boldest reduction rho (smallest mu' = rho mu) that keeps
    x' > 0, s' > 0 and sigma^2 <= bound. rho = 1 - 1/(6 sqrt m) is always safe.
    """
    m = A.shape[1]
    rho_safe = 1.0 - 1.0 / (6.0 * np.sqrt(m))
    lo, hi = 0.0, rho_safe                   # lo = bold (maybe unsafe), hi = safe

    def safe(rho):
        x2, y2, s2 = ip_step(A, b, x, y, s, rho * mu)
        return (x2 > 0).all() and (s2 > 0).all() and sigma_sq(x2, s2, rho * mu) <= bound

    for _ in range(tries):
        mid = 0.5 * (lo + hi)
        if safe(mid):
            hi = mid                          # works, try bolder
        else:
            lo = mid                          # too bold, pull back
    return hi


def ip_solve_adaptive(A, b, c, x, y, s, mu, mu_tol=1e-9, max_iter=2000):
    """Adaptive-step iteration. Returns final (x, y, s, mu) and history."""
    hist = {"it": [], "gap": [], "primal": [], "dual": []}
    for it in range(max_iter):
        hist["it"].append(it)
        hist["gap"].append(float(x @ s))
        hist["primal"].append(float(c @ x))
        hist["dual"].append(float(b @ y))
        if mu < mu_tol:
            break
        rho = adaptive_factor(A, b, x, y, s, mu)
        mu = rho * mu
        x, y, s = ip_step(A, b, x, y, s, mu)
    return x, y, s, mu, hist


# ----------------------------------------------------------------------
# Problem 10: rounding decision (set the smaller of x_i, s_i to zero)
# ----------------------------------------------------------------------
def round_decisions(x, s):
    """Return a list: 'x' if x_i -> 0 (off), 's' if s_i -> 0 (active)."""
    return ['x' if x[i] < s[i] else 's' for i in range(len(x))]


def print_rounding(x, s):
    dec = round_decisions(x, s)
    print("  rounding (smaller of x_i, s_i -> 0):")
    for i, d in enumerate(dec):
        if d == 'x':
            print(f"    i={i+1}: x{i+1} = 0   ({x[i]:.2e} < {s[i]:.2e})   OFF")
        else:
            print(f"    i={i+1}: s{i+1} = 0   ({s[i]:.2e} < {x[i]:.2e})   ACTIVE")
    return dec


# ----------------------------------------------------------------------
# Problem 11: exact rational solve from the rounding, then verify
# ----------------------------------------------------------------------
def exact_solve(A, b, c, decisions):
    """
    Impose the m rounding equations on  A x = b,  A^T y + s = c  and solve
    exactly (rational arithmetic). Return (x, y, s) and a dict of checks.
    """
    import sympy as sp
    Am = sp.Matrix([[sp.nsimplify(v) for v in row] for row in A.tolist()])
    bm = sp.Matrix([sp.nsimplify(v) for v in b.tolist()])
    cm = sp.Matrix([sp.nsimplify(v) for v in c.tolist()])
    n, m = Am.shape

    x = sp.Matrix(sp.symbols(f'x1:{m+1}', real=True))
    y = sp.Matrix(sp.symbols(f'y1:{n+1}', real=True))
    s = sp.Matrix(sp.symbols(f's1:{m+1}', real=True))

    eqs  = [sp.Eq((Am * x)[i], bm[i]) for i in range(n)]          # A x = b
    eqs += [sp.Eq((Am.T * y)[j] + s[j], cm[j]) for j in range(m)]  # A^T y + s = c
    for j in range(m):                                            # rounding eqs
        eqs.append(sp.Eq(x[j], 0) if decisions[j] == 'x' else sp.Eq(s[j], 0))

    sol = sp.solve(eqs, list(x) + list(y) + list(s), dict=True)[0]
    xv = sp.Matrix([sol[v] for v in x])
    yv = sp.Matrix([sol[v] for v in y])
    sv = sp.Matrix([sol[v] for v in s])

    checks = {
        "primal_feasible": (Am * xv == bm) and all(v >= 0 for v in xv),
        "dual_feasible":   (Am.T * yv + sv == cm) and all(v >= 0 for v in sv),
        "gap":             sp.simplify((cm.T * xv - bm.T * yv)[0]),
        "cost":            sp.simplify((cm.T * xv)[0]),
        "complementary":   all(sp.simplify(xv[i] * sv[i]) == 0 for i in range(m)),
    }
    return xv, yv, sv, checks


# ----------------------------------------------------------------------
# Problem 12: reference solve with HiGHS (simplex and interior point)
# ----------------------------------------------------------------------
def solve_with_highs(A, b, c):
    """Solve with HiGHS dual simplex and interior point via scipy.linprog."""
    from scipy.optimize import linprog
    bounds = [(0, None)] * A.shape[1]
    out = {}
    for method in ("highs-ds", "highs-ipm"):
        r = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method=method)
        out[method] = (r.x, float(r.fun), int(r.nit))
    return out


# ----------------------------------------------------------------------
# Benchmark: iterations, runtime, error across all methods
# ----------------------------------------------------------------------
def benchmark(A, b, c, start, optimum, repeats=30):
    import time
    from scipy.optimize import linprog

    def timed(fn):
        best, out = float("inf"), None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return out, best

    rows = []
    (out, t) = timed(lambda: ip_solve(A, b, c, *start, mu_tol=1e-9))
    x = out[0]
    rows.append(("Own IP (fixed)", out[4]["it"][-1], t, abs(float(c @ x) - optimum)))

    (out, t) = timed(lambda: ip_solve_adaptive(A, b, c, *start, mu_tol=1e-9))
    x = out[0]
    rows.append(("Own IP (adaptive)", out[4]["it"][-1], t, abs(float(c @ x) - optimum)))

    bounds = [(0, None)] * A.shape[1]
    for method, name in [("highs-ds", "HiGHS simplex"), ("highs-ipm", "HiGHS interior")]:
        r, t = timed(lambda mm=method: linprog(c, A_eq=A, b_eq=b, bounds=bounds, method=mm))
        rows.append((name, r.nit, t, abs(float(r.fun) - optimum)))

    print(f"\n{'method':<20}{'iters':>8}{'runtime (ms)':>16}{'|cTx - opt|':>16}")
    print("-" * 60)
    for name, it, t, d in rows:
        print(f"{name:<20}{it:>8}{t*1e3:>16.3f}{d:>16.2e}")
    return rows


# ----------------------------------------------------------------------
# Convergence plots (fixed on top, adaptive below; shared x-axis)
# ----------------------------------------------------------------------
def plot_convergence(hist, hist_adaptive, optimum, fname="ip_convergence.png"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    xmax = hist["it"][-1]

    def gap_panel(a, h, color, label):
        a.semilogy(h["it"], h["gap"], color=color, lw=2, label=label)
        a.set_title("Duality Gap Convergence (log scale)")
        a.set_xlabel("Iteration")
        a.set_ylabel("Duality gap  $x^{T}s$")
        a.set_xlim(0, xmax)
        a.legend(loc="best", frameon=True)
        a.grid(True, which="major", alpha=0.3)

    def obj_panel(a, h):
        a.plot(h["it"], h["primal"], color="#16a34a", lw=2, label="Primal  $c^{T}x$")
        a.plot(h["it"], h["dual"],   color="#dc2626", lw=2, label="Dual  $b^{T}y$")
        a.axhline(optimum, color="gray", ls="--", lw=1, label="Optimum  $-44/9$")
        a.set_title("Primal and Dual Objective Values")
        a.set_xlabel("Iteration")
        a.set_ylabel("Objective value")
        a.set_xlim(0, xmax)
        a.legend(loc="best", frameon=True)
        a.grid(True, alpha=0.3)

    gap_panel(ax[0, 0], hist, "#2563eb", "Fixed step")
    obj_panel(ax[0, 1], hist)
    gap_panel(ax[1, 0], hist_adaptive, "#ea580c", "Adaptive step")
    obj_panel(ax[1, 1], hist_adaptive)

    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    return fname


# ----------------------------------------------------------------------
# Run on problem X
# ----------------------------------------------------------------------
if __name__ == "__main__":
    A = np.array([[3., 3., 3., 0., 0.],
                  [3., 1., 0., 1., 0.],
                  [1., 4., 0., 0., 1.]])
    b = np.array([4., 3., 4.])
    c = np.array([-3., -4., 0., 0., 0.])
    OPT = -44 / 9

    def start():
        return (np.array([2/5, 8/15, 2/5, 19/15, 22/15]),
                np.array([-4/5, -4/5, -2/3]),
                np.array([37/15, 28/15, 12/5, 4/5, 2/3]),
                1.0)

    # problem 7: check the starting point satisfies the invariant
    x0, y0, s0, mu0 = start()
    print(f"start: sigma^2 = {sigma_sq(x0, s0, mu0):.6f}  (bound 1/36 = {1/36:.6f})")

    # problem 8: fixed step
    x, y, s, mu, hist = ip_solve(A, b, c, *start(), mu_tol=1e-9)
    print(f"\nfixed step: {hist['it'][-1]} iterations, mu = {mu:.2e}")
    print("  x =", np.round(x, 6))
    print(f"  c^T x = {c @ x:.6f}   b^T y = {b @ y:.6f}")
    print_rounding(x, s)

    # problem 9: adaptive step
    xa, ya, sa, mua, hist_a = ip_solve_adaptive(A, b, c, *start(), mu_tol=1e-9)
    print(f"\nadaptive step: {hist_a['it'][-1]} iterations, mu = {mua:.2e}")
    print(f"  c^T x = {c @ xa:.6f}   b^T y = {b @ ya:.6f}")

    # convergence figure
    plot_convergence(hist, hist_a, optimum=OPT)
    print("\nsaved ip_convergence.png")

    # problem 11: exact rational solve and verification
    print("\nexact solve (problem 11):")
    xv, yv, sv, checks = exact_solve(A, b, c, round_decisions(x, s))
    print("  x* =", list(xv))
    print("  y* =", list(yv))
    print("  s* =", list(sv))
    print("  primal feasible:", checks["primal_feasible"],
          " dual feasible:", checks["dual_feasible"])
    print("  duality gap:", checks["gap"], "  cost c^T x =", checks["cost"])
    print("  complementary:", checks["complementary"])

    # problem 12: HiGHS reference
    print("\nHiGHS reference solve (problem 12):")
    for method, (xr, opt, nit) in solve_with_highs(A, b, c).items():
        label = "simplex " if method == "highs-ds" else "interior"
        print(f"  {label} ({method}): x = {np.round(xr, 6)}, optimum = {opt:.6f}, iters = {nit}")

    # benchmark table
    benchmark(A, b, c, start(), optimum=OPT)
