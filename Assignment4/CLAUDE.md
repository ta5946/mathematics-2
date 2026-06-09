# MAT2 Homework 4.1

## What this is

Solution to MAT2 Part 4 Homework 1, covering Nelder-Mead optimization, comparison with gradient methods, black-box optimization, and a toy LP. Everything lives in `nm_method.ipynb`. The executable `hw_4_1_win.exe` is required for the black-box section.

Student ID: `63210005`

---

## Notebook sections

### 1. Nelder-Mead implementation
2D, 3D, and n-dimensional versions. The n-dimensional `nelder_mead` is the general one — the 2D and 3D versions exist because the homework explicitly asks for them.

### 2. Comparison with gradient methods
NM vs GD, Polyak, Nesterov, AdaGrad, Newton, BFGS on three test functions. Five comparison criteria: information used, convergence speed, convergence quality, sensitivity to starting point, sensitivity to parameters.

### 3. Black-box optimization
Two methods run against the black-box executable: `nelder_mead_bb` and `bfgs_bb`.

**Important:** the BB run loops are commented out and results are hardcoded. Each full run takes ~12 minutes (6 functions × 2 minutes). To re-run, uncomment the loop block and comment out the hardcoded dict in each cell.

`grad_fd` is sequential on purpose — parallel finite differences made BFGS appear to do fewer iterations than NM on f2, breaking the fairness of the comparison.

BFGS struggles on f2 because the finite-difference gradients are misleading near that function's landscape, causing ~9 Armijo halvings per iteration. Only 3 iterations complete within the time limit.

The time limit check fires at the start of each loop iteration, so the last iteration always runs to completion. Actual elapsed time is 130-160s per function despite a 120s limit.

### 4. Toy LP
Enumerate all C(8,3) = 56 triples of constraints, solve each as a 3x3 linear system, keep feasible solutions, pick the one maximizing c @ x. Optimal: x* = (0, 2.5, 1), f = 11.5.

---

## Key decisions

- BFGS parameters (c1=1e-4, backtrack=0.5, lr_min=1e-16, curvature_min=1e-10) are standard Nocedal & Wright and match the `bfgs` implementation in the comparison section.
- Tolerances are consistent across paired methods: both `nelder_mead_bb` and `bfgs_bb` use `tol=1e-12`.
- No comments explaining what the code does — only comments explaining why non-obvious decisions were made.
