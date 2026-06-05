## Implementing Nelder-Mead method

**Question.** Why fix the dimension? Surely you can pass the dimension as a parameter or better still compute it on the fly.

**Answer.** There is no real reason to fix the dimension. Nelder-Mead works in dimension `n` by using `n + 1` points, so the code can just compute the dimension from the starting point.

In my implementation I use:

```python
n = len(x0)
```

Then I create the starting simplex from `x0` and small shifts in each coordinate direction:

```python
x0,
x0 + d * e1,
x0 + d * e2,
...
x0 + d * en
```

Here `d` is the chosen diameter and `e1, ..., en` are the coordinate unit vectors. So in 2D the method uses 3 points, in 3D it uses 4 points, and the same function would also work for higher dimensions. The only downside is that Nelder-Mead usually becomes slower in higher dimensions because the simplex has more points.

## Comparing Nelder-Mead with gradient descent methods

**Question.** Qualitatively compare?? Think of all possible aspects of comparison. The word all in the previous sentence is maybe too ambitious, several is a more appropriate choice.

**Answer.** The methods can be compared using several different criteria:

1. **Information used**

   Nelder-Mead only uses function values. It does not need gradients or Hessians. GD, Polyak GD, Nesterov GD, and AdaGrad use gradients. Newton uses gradients and Hessians, while BFGS uses gradients and builds an approximation of second-order information.

2. **Convergence iterations**

   This means how many iterations the method needs before it gets close to a minimum. Newton and BFGS often need fewer iterations. Basic GD usually needs more iterations, especially on functions with narrow valleys. Nelder-Mead can also need many iterations because it searches only using function values.

3. **Convergence quality**

   This compares the final function value and final point. A method is better if it reaches a lower function value and gets closer to the expected minimum. Some methods may move quickly at first but then stagnate, while others may converge more slowly but reach a better final value.

4. **Evaluations per second**

   This compares how much useful work the method can do in a fixed amount of time. For Nelder-Mead, this mainly means function evaluations per second. For gradient methods, gradient evaluations are also important. This is useful because comparing only the number of iterations is not always fair.

5. **Dependence on starting point**

   All methods can depend on the starting point because they are local optimization methods. This is especially visible on non-quadratic functions such as the Rosenbrock-like function and the Beale function. Nelder-Mead also depends on the initial simplex, not only on one starting point.

6. **Sensitivity to parameters**

   GD depends strongly on the learning rate. Polyak GD and Nesterov GD also depend on the momentum parameter. AdaGrad depends on the initial learning rate. Newton may need damping or regularization if the Hessian causes problems. Nelder-Mead depends mostly on the initial simplex diameter and its reflection, expansion, contraction, and shrink parameters.

7. **Generality across functions**

   A method is more general if it behaves reasonably on all three test functions. Nelder-Mead is general because it only needs function values, so it can be used even when derivatives are not available. Gradient-based methods can be faster when derivatives are known, but they may be less convenient or less stable on difficult functions.

### Test functions

The comparison was done on the same three functions as in Homework GD.2.

1. Quadratic function in three variables:

   $$f_a(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y.$$

   Starting points: `(0,0,0)` and `(1,1,0)`.

2. Rosenbrock-like function in three variables:

   $$f_b(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2.$$

   Starting points: `(1.2,1.2,1.2)` and `(-1,1.2,1.2)`.

3. Beale function in two variables:

   $$f_c(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2.$$

   Starting points: `(1,1)` and `(4.5,4.5)`.

For the gradient-based methods, the gradients and Hessians from the previous homework were used. For Nelder-Mead, the same starting points were used as one vertex of the initial simplex, and the simplex diameter was varied.

### Numerical results

The comparison was run with `max_iter = 1000`. The gradient-based methods used the same parameters as in the previous homework:

- GD: `lr = 0.001`
- Polyak GD: `lr = 0.001`, `momentum = 0.9`
- Nesterov GD: `lr = 0.001`, `momentum = 0.9`
- AdaGrad: `lr = 0.01`
- Newton: `step_scale = 1.0`
- BFGS: Armijo line search
- Nelder-Mead: simplex diameters `0.1`, `1.0`, and `3.0`

The best final values from the code output were:

| function | starting point | best method | final value | final point |
|---|---:|---|---:|---|
| `f_a` | `(0,0,0)` | Newton | `-0.197916666667` | `(-0.166667, -0.229167, 0.166667)` |
| `f_a` | `(1,1,0)` | Polyak GD | `-0.197916666667` | `(-0.166667, -0.229167, 0.166667)` |
| `f_b` | `(1.2,1.2,1.2)` | Newton | `6.71e-25` | `(1, 1, 1)` |
| `f_b` | `(-1,1.2,1.2)` | Newton | `2.99e-19` | `(1, 1, 1)` |
| `f_c` | `(1,1)` | Nelder-Mead, `d=0.1` | `9.98e-19` | `(3, 0.5)` |
| `f_c` | `(4.5,4.5)` | Nelder-Mead, `d=1.0` | `8.87e-19` | `(3, 0.5)` |

For the quadratic function `f_a`, Newton reached the minimum in one step, as expected for a quadratic function with an exact Hessian. BFGS also reached the same minimum in a small number of iterations. Nelder-Mead also found the correct minimum for all tested simplex diameters, but it needed more iterations because it only uses function values.

For the Rosenbrock-like function `f_b`, Newton gave the best final values from both starting points. Nelder-Mead also reached the minimum accurately, but required more iterations. The first-order methods were much slower with the fixed parameters from the previous homework.

For the Beale function `f_c`, Nelder-Mead performed best from both starting points. This was especially clear from the difficult starting point `(4.5,4.5)`, where GD, Polyak GD, and Nesterov GD diverged. This shows that Nelder-Mead can be useful on difficult non-quadratic functions when a derivative-based method is sensitive to the starting point or step size.

### Findings by comparison criterion

1. **Information used**

   Nelder-Mead used only function values. This makes it the most general method in terms of required input. Newton used the most information, because it required both gradients and Hessians.

2. **Convergence iterations**

   Newton needed the fewest iterations on `f_a` and `f_b`. BFGS also converged quickly on `f_a` and `f_c` from `(1,1)`. Nelder-Mead usually needed more iterations, but still converged reliably on all tested functions.

3. **Convergence quality**

   Newton was best on the quadratic and Rosenbrock-like functions. Nelder-Mead was best on the Beale function. All tested Nelder-Mead diameters reached almost the same minimum on `f_a` and `f_c`, while the first-order methods sometimes stopped far from the minimum or diverged.

4. **Evaluations per second**

   The gradient-based methods had more evaluations per second in this implementation, because the objective functions and derivatives are cheap to compute. Nelder-Mead had fewer evaluations per second, but each evaluation only required the function value.

5. **Dependence on starting point**

   The starting point mattered most for `f_b` and `f_c`. On `f_c`, the starting point `(4.5,4.5)` caused GD, Polyak GD, and Nesterov GD to diverge, while Nelder-Mead still found the minimum.

6. **Sensitivity to parameters**

   For Nelder-Mead, this was tested directly by using simplex diameters `0.1`, `1.0`, and `3.0`. All three worked reasonably well, but the best diameter was not always the same for every function. For the gradient-based methods, the parameters were kept the same as in the previous homework, so their parameter sensitivity was discussed qualitatively rather than tested again.

7. **Generality across functions**

   Newton was very strong when derivatives and Hessians worked well. BFGS was also strong, especially when the gradient gave useful curvature information. Nelder-Mead was the most generally applicable method because it only used function values and still performed well on all three functions.

## Black box optimization

The student ID used for this part is `63210005`. Therefore the three black-box functions are maps

$$f_{63210005,i}: \mathbb{R}^3 \to \mathbb{R}, \qquad i \in \{1,2,3\}.$$

They are evaluated through the provided executable:

```text
hw_4_1_win.exe 63210005 i x y z
```

where `i` is one of `1`, `2`, or `3`, and `(x,y,z)` is the point where the function is evaluated.

**Question.** How would one use a gradient-descent based method in such a case? Which one is best suitable? Can you beat Nelder-Mead?

**Answer.** Since the functions are black boxes, exact gradients are not available. A gradient-based method can still be used by approximating the gradient with two-sided finite differences:

$$
\frac{\partial f}{\partial x_j}(x) \approx
\frac{f(x + h e_j) - f(x - h e_j)}{2h}.
$$

Since the functions are maps from `R^3` to `R`, one such gradient approximation needs `6` black-box evaluations. The most suitable gradient-based method is BFGS, because it uses gradient information efficiently and builds approximate second-order information without needing an exact Hessian. The result can then be compared with Nelder-Mead to check whether the gradient-based method gives a lower value.

### Black-box numerical results

The following run used BFGS with central finite-difference gradients and Nelder-Mead, both from the starting point `(0,0,0)` with `max_iter = 20`.

```text
============================================================
Black-box function f_63210005,1
============================================================
BFGS with finite-difference gradient   start=[0, 0, 0] f=0.50001236, x=[0.36665, 0.123667, 0.00123667], calls=36, time=33.2s
Nelder-Mead d=1                        start=[0, 0, 0] f=0.500104356666149, x=[0.375927, 0.125327, 0.00301725], calls=43, time=124.7s

Best candidate:
BFGS with finite-difference gradient, f=0.50001236, x=[0.36665, 0.123667, 0.00123667], calls=36, time=33.2s

============================================================
Black-box function f_63210005,2
============================================================
BFGS with finite-difference gradient   start=[0, 0, 0] f=0.509324777737146, x=[0.0987744, -0.032654, 0.396529], calls=172, time=192.3s
Nelder-Mead d=1                        start=[0, 0, 0] f=0.723936815252721, x=[0.350221, -0.383783, 0.484571], calls=40, time=115.7s

Best candidate:
BFGS with finite-difference gradient, f=0.509324777737146, x=[0.0987744, -0.032654, 0.396529], calls=172, time=192.3s

============================================================
Black-box function f_63210005,3
============================================================
BFGS with finite-difference gradient   start=[0, 0, 0] f=0.500012360000003, x=[0.123666, 0.00123679, 0.36665], calls=70, time=57.4s
Nelder-Mead d=1                        start=[0, 0, 0] f=0.500049027835165, x=[0.120123, -0.00273323, 0.356297], calls=42, time=123.7s

Best candidate:
BFGS with finite-difference gradient, f=0.500012360000003, x=[0.123666, 0.00123679, 0.36665], calls=70, time=57.4s
```

With this iteration budget, BFGS with finite-difference gradients gave a lower value than Nelder-Mead on all three black-box functions. The results for the first and third functions appear stable, while the second function may still require additional refinement if twelve significant digits are required.

## A toy solution to a linear problem

There are 3 variables and 8 inequalities, including the three nonnegativity constraints. The method checks all triples of active constraints:

$$\binom{8}{3} = 56.$$

For each triple, the three selected inequalities are treated as equalities. The resulting linear system is solved, and the solution is kept only if it satisfies all 8 original inequalities.

Since the problem is a maximization problem, the feasible candidate with the largest value of `x1 + 3x2 + 4x3` is selected.

The code found 6 feasible vertices. The optimal solution is

$$x = (0, 2.5, 1).$$

The objective value is

$$x_1 + 3x_2 + 4x_3 = 0 + 3 \cdot 2.5 + 4 \cdot 1 = 11.5.$$

The active constraints at the optimum are:

```text
x1 + 2x2 + 3x3 = 8
3x1 + 2x2 + x3 = 6
x1 = 0
```

**Question.** How does this approach generalize if we increase the number of variables and the number of constraints?

**Answer.** If there are `n` variables and `m` inequalities, then a vertex is usually determined by `n` active constraints. The same method would check all

$$\binom{m}{n}$$

choices of active constraints. This works for small problems, but becomes too expensive when `m` and `n` grow. For larger LPs, simplex or interior-point methods are more appropriate.
