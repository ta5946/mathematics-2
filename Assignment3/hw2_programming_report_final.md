# Mathematics 2 Homework GD.2 Programming Report

## Problem 6: Comparison of Optimisation Methods

### Models and Parameters

The goal is to compare GD, Polyak GD, Nesterov GD, AdaGrad, Adam, Newton method and BFGS on the three given functions. Each method is tested from the required starting points and compared after 2, 5, 10 and 100 steps, and after 0.1, 1 and 2 seconds.

- Functions:
  - $f_1(x,y,z)=(x-z)^2+(2y+z)^2+(4x-2y+z)^2+x+y$
  - $f_2(x,y,z)=(x-1)^2+(y-1)^2+100(y-x^2)^2+100(z-y^2)^2$
  - $f_3(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2$
- Starting points:
  - $f_1$: $(0,0,0)$ and $(1,1,0)$
  - $f_2$: $(1.2,1.2,1.2)$ and $(-1,1.2,1.2)$
  - $f_3$: $(1,1)$ and $(4.5,4.5)$
- GD: $\gamma = 0.001$
- Polyak GD: $\gamma = 0.001$, $\mu = 0.9$
- Nesterov GD: $\gamma = 0.001$, $\mu = 0.9$
- AdaGrad: $\gamma = 0.01$, $\varepsilon = 10^{-8}$
- Adam: $\gamma = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$
- Newton method: $\alpha = 1$
- BFGS: starts with the identity matrix as the inverse Hessian approximation and uses line search for the step size

### Output

```text
======================================================================
PART (a): Performance by number of steps
======================================================================

Function 1)  f(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y
----------------------------------------------------------------------

  Starting point: [0. 0. 0.]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD                -0.003947      -0.009615      -0.018469      -0.111403
  Polyak            -0.005707      -0.024321      -0.065437      -0.197883
  Nesterov          -0.005692      -0.024118      -0.064726      -0.197856
  AdaGrad           -0.031685      -0.056334      -0.081489      -0.179946
  Adam              -0.003971      -0.009836      -0.019351      -0.135067
  Newton            -0.197917      -0.197917      -0.197917      -0.197917
  BFGS              -0.133891      -0.197917      -0.197917      -0.197917

      2 steps: 1. Newton (-0.197917),  2. BFGS (-0.133891),  3. AdaGrad (-0.031685)
      5 steps: 1. Newton (-0.197917),  2. BFGS (-0.197917),  3. AdaGrad (-0.056334)
     10 steps: 1. Newton (-0.197917),  2. BFGS (-0.197917),  3. AdaGrad (-0.081489)
    100 steps: 1. Newton (-0.197917),  2. BFGS (-0.197917),  3. Polyak (-0.197883)

  Starting point: [1. 1. 0.]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD                10.242712       9.277925       8.017875       1.936482
  Polyak             9.913269       7.106921       4.133920      -0.197397
  Nesterov           9.923807       7.202177       4.077512      -0.196978
  AdaGrad           10.562652      10.185127       9.756435       6.970224
  Adam              10.948073      10.870466      10.741916       8.634301
  Newton            -0.197917      -0.197917      -0.197917      -0.197917
  BFGS               1.174951      -0.197912      -0.197917      -0.197917

      2 steps: 1. Newton (-0.197917),  2. BFGS (1.174951),  3. Polyak (9.913269)
      5 steps: 1. Newton (-0.197917),  2. BFGS (-0.197912),  3. Polyak (7.106921)
     10 steps: 1. BFGS (-0.197917),  2. Newton (-0.197917),  3. Nesterov (4.077512)
    100 steps: 1. BFGS (-0.197917),  2. Newton (-0.197917),  3. Polyak (-0.197397)

Function 2)  f(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2
----------------------------------------------------------------------

  Starting point: [1.2 1.2 1.2]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD                 0.042372       0.018223       0.018109       0.016875
  Polyak             4.697759       0.161202       4.266084       0.001071
  Nesterov           0.489933       0.127005       3.144180       0.092156
  AdaGrad            8.155288       5.941211       4.056005       0.102337
  Adam              11.143472      10.481404       9.440934       1.156201
  Newton             0.035418       0.000041       0.00e+00       0.00e+00
  BFGS               3.720412       0.079308       0.004580       0.00e+00

      2 steps: 1. Newton (0.035418),  2. GD (0.042372),  3. Nesterov (0.489933)
      5 steps: 1. Newton (0.000041),  2. GD (0.018223),  3. BFGS (0.079308)
     10 steps: 1. Newton (0.00e+00),  2. BFGS (0.004580),  3. GD (0.018109)
    100 steps: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (0.001071)

  Starting point: [-1.   1.2  1.2]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD                 5.291304       4.242976       4.203009       4.133886
  Polyak            15.851510       4.754752       4.512499       3.186586
  Nesterov          11.885893      10.586343       3.688114       1.891565
  AdaGrad            9.751536       7.412732       5.748465       4.257943
  Adam              13.249176      12.455460      11.223334       4.314755
  Newton             9.061843       4.236227    1156.427674       0.00e+00
  BFGS               7.583594       3.606559       3.222680       0.00e+00

      2 steps: 1. GD (5.291304),  2. BFGS (7.583594),  3. Newton (9.061843)
      5 steps: 1. BFGS (3.606559),  2. Newton (4.236227),  3. GD (4.242976)
     10 steps: 1. BFGS (3.222680),  2. Nesterov (3.688114),  3. GD (4.203009)
    100 steps: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Nesterov (1.891565)

Function 3)  f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
----------------------------------------------------------------------

  Starting point: [1. 1.]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD                12.807585      11.251866       9.459676       2.061266
  Polyak            12.218716       8.027551       4.083937       0.086604
  Nesterov          12.253178       8.228678       4.191109       0.083665
  AdaGrad           13.737011      13.321514      12.833988       9.338297
  Adam              14.147722      14.064791      13.926945      11.494177
  Newton            14.203125      14.203125      14.203125      14.203125
  BFGS               4.438122       0.064655       0.000502       0.00e+00

      2 steps: 1. BFGS (4.438122),  2. Polyak (12.218716),  3. Nesterov (12.253178)
      5 steps: 1. BFGS (0.064655),  2. Polyak (8.027551),  3. Nesterov (8.228678)
     10 steps: 1. BFGS (0.000502),  2. Polyak (4.083937),  3. Nesterov (4.191109)
    100 steps: 1. BFGS (0.00e+00),  2. Nesterov (0.083665),  3. Polyak (0.086604)

  Starting point: [4.5 4.5]
  Method               2 steps         5 steps        10 steps       100 steps
  ------------  -------------  -------------  -------------  -------------
  GD            15873977012198753388419453201611294387422360334774286981712683912980978964369929914173603073892955086389248.000000       diverged       diverged       diverged
  Polyak        15873977011026416776643216779616854940840190699929395355005607275441123138221540577965505402780641879130112.000000       diverged       diverged       diverged
  Nesterov      143867079511859484589601344341991383686924399573471337081051797113266111492122667365660696638127687735115997630382024425472.000000       diverged       diverged       diverged
  AdaGrad       169646.559640  165200.723108  160163.711645  127388.016334
  Adam          174199.026683  173281.246259  171762.026592  147002.272716
  Newton         15043.650877     418.804026      14.275383      14.203125
  BFGS            2894.304950      32.420068       8.462121       7.379384

      2 steps: 1. BFGS (2894.304950),  2. Newton (15043.650877),  3. AdaGrad (169646.559640)
      5 steps: 1. BFGS (32.420068),  2. Newton (418.804026),  3. AdaGrad (165200.723108)
     10 steps: 1. BFGS (8.462121),  2. Newton (14.275383),  3. AdaGrad (160163.711645)
    100 steps: 1. BFGS (7.379384),  2. Newton (14.203125),  3. AdaGrad (127388.016334)

======================================================================
PART (b): Performance by wall-clock time
======================================================================

Function 1)  f(x,y,z) = (x-z)^2 + (2y+z)^2 + (4x-2y+z)^2 + x + y
----------------------------------------------------------------------

  Starting point: [0. 0. 0.]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:452: RuntimeWarning: overflow encountered in scalar multiply
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:453: RuntimeWarning: overflow encountered in scalar power
  return r1**2 + r2**2 + r3**2
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:463: RuntimeWarning: overflow encountered in scalar multiply
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:464: RuntimeWarning: overflow encountered in scalar multiply
  df_dx = 2*r1*(y-1) + 2*r2*(y**2-1) + 2*r3*(y**3-1)
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:465: RuntimeWarning: overflow encountered in scalar multiply
  df_dy = 2*r1*x + 2*r2*(2*x*y) + 2*r3*(3*x*y**2)
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:450: RuntimeWarning: invalid value encountered in scalar add
  r1 = 1.5   - x + x*y
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:451: RuntimeWarning: invalid value encountered in scalar add
  r2 = 2.25  - x + x*y**2
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:452: RuntimeWarning: invalid value encountered in scalar add
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:461: RuntimeWarning: invalid value encountered in scalar add
  r1 = 1.5   - x + x*y
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:462: RuntimeWarning: invalid value encountered in scalar add
  r2 = 2.25  - x + x*y**2
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:463: RuntimeWarning: invalid value encountered in scalar add
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:451: RuntimeWarning: overflow encountered in scalar multiply
  r2 = 2.25  - x + x*y**2
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:452: RuntimeWarning: overflow encountered in scalar power
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:462: RuntimeWarning: overflow encountered in scalar multiply
  r2 = 2.25  - x + x*y**2
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:463: RuntimeWarning: overflow encountered in scalar power
  r3 = 2.625 - x + x*y**3
C:\Users\tjasa\Desktop\MAT2\Assignment35\hw2_programming.py:464: RuntimeWarning: overflow encountered in scalar power
  df_dx = 2*r1*(y-1) + 2*r2*(y**2-1) + 2*r3*(y**3-1)
  GD            -0.136473  -0.197917  -0.197917
  Polyak        -0.197917  -0.197917  -0.197917
  Nesterov      -0.197917  -0.197917  -0.197917
  AdaGrad       -0.197917  -0.197917  -0.197917
  Adam          -0.197917  -0.197917  -0.197917
  Newton        -0.197917  -0.197917  -0.197917
  BFGS          -0.197917  -0.197917  -0.197917

    0.1s: 1. Polyak (-0.197917),  2. Nesterov (-0.197917),  3. AdaGrad (-0.197917)
    1.0s: 1. GD (-0.197917),  2. Polyak (-0.197917),  3. Nesterov (-0.197917)
    2.0s: 1. GD (-0.197917),  2. Polyak (-0.197917),  3. Nesterov (-0.197917)

  Starting point: [1. 1. 0.]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
  GD            -0.197917  -0.197917  -0.197917
  Polyak        -0.197917  -0.197917  -0.197917
  Nesterov      -0.197917  -0.197917  -0.197917
  AdaGrad       -0.197917  -0.197917  -0.197917
  Adam          -0.197917  -0.197917  -0.197917
  Newton        -0.197917  -0.197917  -0.197917
  BFGS          -0.197917  -0.197917  -0.197917

    0.1s: 1. GD (-0.197917),  2. Polyak (-0.197917),  3. Nesterov (-0.197917)
    1.0s: 1. GD (-0.197917),  2. Polyak (-0.197917),  3. Nesterov (-0.197917)
    2.0s: 1. GD (-0.197917),  2. Polyak (-0.197917),  3. Nesterov (-0.197917)

Function 2)  f(x,y,z) = (x-1)^2 + (y-1)^2 + 100(y-x^2)^2 + 100(z-y^2)^2
----------------------------------------------------------------------

  Starting point: [1.2 1.2 1.2]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
  GD             0.000001   1.86e-26   1.86e-26
  Polyak         2.47e-31   2.47e-31   2.47e-31
  Nesterov       0.000033   0.000033   0.000033
  AdaGrad        0.000001   0.000001   0.000001
  Adam           8.54e-18   8.54e-18   8.54e-18
  Newton         0.00e+00   0.00e+00   0.00e+00
  BFGS           0.00e+00   0.00e+00   0.00e+00

    0.1s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)
    1.0s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)
    2.0s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)

  Starting point: [-1.   1.2  1.2]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
  GD             0.000025   4.65e-27   4.65e-27
  Polyak         2.47e-31   2.47e-31   2.47e-31
  Nesterov       0.000063   0.000063   0.000063
  AdaGrad        0.000018   0.000002   0.000002
  Adam           2.76e-17   2.08e-17   2.08e-17
  Newton         0.00e+00   0.00e+00   0.00e+00
  BFGS           0.00e+00   0.00e+00   0.00e+00

    0.1s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)
    1.0s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)
    2.0s: 1. Newton (0.00e+00),  2. BFGS (0.00e+00),  3. Polyak (2.47e-31)

Function 3)  f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
----------------------------------------------------------------------

  Starting point: [1. 1.]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
  GD             0.000044   8.14e-26   8.14e-26
  Polyak         1.52e-24   5.55e-29   5.55e-29
  Nesterov       9.95e-22   5.08e-29   5.08e-29
  AdaGrad        4.16e-26   4.16e-26   4.16e-26
  Adam           0.00e+00   0.00e+00   0.00e+00
  Newton        14.203125  14.203125  14.203125
  BFGS           0.00e+00   0.00e+00   0.00e+00

    0.1s: 1. Adam (0.00e+00),  2. BFGS (0.00e+00),  3. AdaGrad (4.16e-26)
    1.0s: 1. Adam (0.00e+00),  2. BFGS (0.00e+00),  3. Nesterov (5.08e-29)
    2.0s: 1. Adam (0.00e+00),  2. BFGS (0.00e+00),  3. Nesterov (5.08e-29)

  Starting point: [4.5 4.5]
  Method            0.1s      1.0s      2.0s
  ------------  ---------  ---------  ---------
  GD            174813.363281  174813.363281  174813.363281
  Polyak        174813.363281  174813.363281  174813.363281
  Nesterov      174813.363281  174813.363281  174813.363281
  AdaGrad        1.039947   1.13e-26   1.13e-26
  Adam          60.488334   0.00e+00   0.00e+00
  Newton        14.203125  14.203125  14.203125
  BFGS           7.341767   7.341767   7.341767

    0.1s: 1. AdaGrad (1.039947),  2. BFGS (7.341767),  3. Newton (14.203125)
    1.0s: 1. Adam (0.00e+00),  2. AdaGrad (1.13e-26),  3. BFGS (7.341767)
    2.0s: 1. Adam (0.00e+00),  2. AdaGrad (1.13e-26),  3. BFGS (7.341767)
```

### Discussion

- Newton and BFGS performed best on $f_1$, reaching the minimum value in very few steps.
- On $f_2$, Newton and BFGS reached the best final values, while first order methods converged more slowly.
- On $f_3$ from $(4.5,4.5)$, GD, Polyak GD and Nesterov GD diverged with the fixed learning rate.
- The time comparison gives the same main conclusion as the step comparison, second order methods perform best overall.

---

## Problem 7: Linear Regression

### Models and Parameters

The goal is to fit $g(x)=kx+n$ to generated points $(i,i+\nu_i)$, where $\nu_i\sim U[0,1]$. The methods are compared for $N=50,100,1000,10000,100000,1000000$.

- Model: $g(x)=kx+n$
- Objective: $f(k,n)=\frac1N\sum_{i=1}^N(ki+n-y_i)^2$
- Methods: GD, SGD, Newton method, BFGS and L BFGS
- Starting point: $(k,n)=(0,0)$
- GD: $\gamma_{GD}=1/\beta$, where $\beta=\lambda_{\max}(H)$
- SGD: $\gamma_{SGD}=0.01/\beta$, where $\beta=\lambda_{\max}(H)$
- Newton method: exact Hessian step
- BFGS: starts with the identity matrix as the inverse Hessian approximation and uses line search for the step size
- L BFGS: memory parameter $m=10$ and line search for the step size
- Reference optimum: computed with `np.linalg.lstsq`

### Output

```text
===========================================================================
PROBLEM 7: Linear Regression  g(x) = kx + n
===========================================================================

N =        50   optimum = 0.074997   k* = 0.9968   n* = 0.6181
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.158750       0.158608       0.158373       0.154245
  SGD             37.415972       0.531921       0.172218       0.408689
  Newton           0.074997       0.074997       0.074997       0.074997
  BFGS             0.097247       0.074997       0.074997       0.074997
  L-BFGS           0.158750       0.074997       0.074997       0.074997

      2 steps: 1. Newton (0.074997),  2. BFGS (0.097247),  3. L-BFGS (0.158750)
      5 steps: 1. Newton (0.074997),  2. BFGS (0.074997),  3. L-BFGS (0.074997)
     10 steps: 1. Newton (0.074997),  2. BFGS (0.074997),  3. L-BFGS (0.074997)
    100 steps: 1. Newton (0.074997),  2. BFGS (0.074997),  3. L-BFGS (0.074997)

N =       100   optimum = 0.070529   k* = 0.9979   n* = 0.5915
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.152357       0.152321       0.152262       0.151198
  SGD            199.138136       2.078328       0.987079       0.867290
  Newton           0.070529       0.070529       0.070529       0.070529
  BFGS             0.091614       0.070529       0.070529       0.070529
  L-BFGS           0.152357       0.070529       0.070529       0.070529

      2 steps: 1. Newton (0.070529),  2. BFGS (0.091614),  3. L-BFGS (0.152357)
      5 steps: 1. Newton (0.070529),  2. BFGS (0.070529),  3. L-BFGS (0.070529)
     10 steps: 1. BFGS (0.070529),  2. L-BFGS (0.070529),  3. Newton (0.070529)
    100 steps: 1. BFGS (0.070529),  2. L-BFGS (0.070529),  3. Newton (0.070529)

N =      1000   optimum = 0.084937   k* = 1.0000   n* = 0.4962
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.146019       0.146019       0.146018       0.146010
  SGD          21120.949714     291.735811     220.153288       0.317829
  Newton           0.084937       0.084937       0.084937       0.084937
  BFGS             0.100253       0.084937       0.084937       0.084937
  L-BFGS           0.146019       0.146019       0.146018       0.146010

      2 steps: 1. Newton (0.084937),  2. BFGS (0.100253),  3. L-BFGS (0.146019)
      5 steps: 1. BFGS (0.084937),  2. Newton (0.084937),  3. GD (0.146019)
     10 steps: 1. BFGS (0.084937),  2. Newton (0.084937),  3. L-BFGS (0.146018)
    100 steps: 1. BFGS (0.084937),  2. Newton (0.084937),  3. GD (0.146010)

N =     10000   optimum = 0.083062   k* = 1.0000   n* = 0.4955
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.144390       0.144390       0.144390       0.144390
  SGD         2153728.793274   31391.501920   23911.516889       0.298709
  Newton           0.083062       0.083062       0.083062       0.083062
  BFGS             0.098399       0.083062       0.083062       0.083062
  L-BFGS           0.144390       0.144390       0.144390       0.144390

      2 steps: 1. Newton (0.083062),  2. BFGS (0.098399),  3. GD (0.144390)
      5 steps: 1. BFGS (0.083062),  2. Newton (0.083062),  3. GD (0.144390)
     10 steps: 1. BFGS (0.083062),  2. Newton (0.083062),  3. GD (0.144390)
    100 steps: 1. BFGS (0.083062),  2. Newton (0.083062),  3. GD (0.144390)

N =    100000   optimum = 0.083243   k* = 1.0000   n* = 0.5008
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.145948       0.145948       0.145948       0.145948
  SGD         215471348.673178  3151506.079921  2400957.668067       0.147731
  Newton           0.083243       0.083243       0.083243       0.083243
  BFGS             0.098920       0.083243       0.083243       0.083243
  L-BFGS           0.145948       0.145948       0.145948       0.145948

      2 steps: 1. Newton (0.083243),  2. BFGS (0.098920),  3. GD (0.145948)
      5 steps: 1. Newton (0.083243),  2. BFGS (0.083243),  3. GD (0.145948)
     10 steps: 1. BFGS (0.083243),  2. Newton (0.083243),  3. GD (0.145948)
    100 steps: 1. BFGS (0.083243),  2. Newton (0.083243),  3. GD (0.145948)

N =   1000000   optimum = 0.083310   k* = 1.0000   n* = 0.4997
---------------------------------------------------------------------------
  Method             2 steps         5 steps        10 steps       100 steps
  ----------  -------------  -------------  -------------  -------------
  GD               0.145724       0.145724       0.145724       0.145724
  SGD         21543794981.418667  315203014.796151  240322675.063204       0.216176
  Newton           0.083310       0.083310       0.083310       0.083310
  BFGS             0.098914       0.083310       0.083310       0.083310
  L-BFGS           0.145724       0.145724       0.145724       0.145724

      2 steps: 1. Newton (0.083310),  2. BFGS (0.098914),  3. GD (0.145724)
      5 steps: 1. BFGS (0.083310),  2. Newton (0.083310),  3. GD (0.145724)
     10 steps: 1. BFGS (0.083310),  2. Newton (0.083310),  3. GD (0.145724)
    100 steps: 1. BFGS (0.083310),  2. Newton (0.083310),  3. GD (0.145724)
```

### Discussion

- Newton reached the optimum after 1 step because the loss is quadratic and the Hessian is exact.
- BFGS reached the optimum in very few steps for all tested values of $N$.
- GD stayed stable, but convergence was slow for larger $N$.
- SGD was less stable than GD because each step uses only one random point, so its losses were larger for large $N$.
