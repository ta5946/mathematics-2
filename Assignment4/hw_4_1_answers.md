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

   The first-order methods depended strongly on the fixed learning rate. Nelder-Mead depended on the simplex diameter, but the tested diameters `0.1`, `1.0`, and `3.0` all worked reasonably well. The best diameter was not always the same for every function.

7. **Generality across functions**

   Newton was very strong when derivatives and Hessians worked well. BFGS was also strong, especially when the gradient gave useful curvature information. Nelder-Mead was the most generally applicable method because it only used function values and still performed well on all three functions.
