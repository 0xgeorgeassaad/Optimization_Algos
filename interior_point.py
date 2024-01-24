import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
import time
from collections import namedtuple


def interior_point(
    A,
    b,
    c,
    eps=1e-4,
    max_iter=1500,
    verbose=True,
    plotting=0,
    solver=0,
    sigma=0.2,
    alpha=0.1,
    case=None
):
  """
  Author: George Assaad (c) 2023
  ----------------------------------
  Linear programming: minimize a linear objective function subject to linear
  equality and inequality constraints.

  Linear programming solves problems of the following form:
      - minimize ::

          c @ x

      - such that ::

          A @ x == b
          0 <= x <= infty

  Parameters
  ----------
  c : 1-D array
      The coefficients of the linear objective function to be minimized.
  A : 2-D array
      The equality constraint matrix. Each row of ``A`` specifies the
      coefficients of a linear equality constraint on ``x``.
  b : 1-D array
      The equality constraint vector. Each element of ``A @ x`` must equal
      the corresponding element of ``b``.
  eps : float
      A tolerance which determines when a residual is "close enough" to
      zero to be considered exactly zero.
  maxiter : int
      Maximum number of iterations to perform.
  disp : bool
        Set to ``True`` to print iteration messages.
        Default: ``False``.
  solver : int
          An integer representing the solver type.

          ``0`` : Adaptive Path following method.

          ``1`` : Fixed size path following method.

          ``2`` : Mehrotra predictor-corrector method.


  Returns
  -------
  x : 1-D array
        The values of the decision variables that minimizes the
        objective function while satisfying the constraints.
  fun : float
        The optimal value of the objective function ``c @ x``.

  References
  ----------
  [1] Dantzig, George B., Linear programming and extensions. Rand
      Corporation Research Study Princeton Univ. Press, Princeton, NJ,
      1963
  [2] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
      In Springer Series in Operations Research and Financial Engineering (pp. 1-664).
      (Springer Series in Operations Research and Financial Engineering). Springer Nature.
  [3] Mehrotra, S. (1992). "On the implementation of a primal–dual interior point method".
      SIAM Journal on Optimization. 2 (4): 575–601. doi:10.1137/0802028
  """
  assert plotting in [0,1,2],  f"plotting is either 0, 1 or 2, got: {plotting}"

  if plotting > 0:
    M = A[:,2:]
    if not ( (M.shape[0] == M.shape[1]) and (M == np.eye(M.shape[0])).all() ):
      print('central path plotting is erroneous, there are more than 2 variables')

  method = 'Adaptive'
  if solver == 1:
    method = 'Fixed'
  elif solver == 2:
    method = 'Mehrotra'

  OptResult = namedtuple("OptResult",["fun","x"])


  # starting point
  x_telda = A.T @ np.linalg.inv(A@A.T) @ b
  y_telda = np.linalg.inv(A@A.T)@A@c
  s_telda = c - A.T@y_telda

  x_telda += max(-3/2 * min(x_telda), 0)
  s_telda += max(-3/2 * min(s_telda), 0)
  x_telda += 1/2 * np.dot(x_telda, s_telda)/sum(s_telda)
  s_telda += 1/2 * np.dot(x_telda, s_telda)/sum(x_telda)
  x_0, y_0, s_0 = x_telda, y_telda, s_telda


  iter = 0
  x_k, y_k, s_k = x_0, y_0, s_0
  fs = [c@x_k]
  xs = [x_k]
  ss = [s_k]
  while (iter < 1 or np.any(np.abs(Delta_x) > eps) or
      np.any(np.abs(Delta_y) > eps) or
      np.any(np.abs(Delta_s) > eps)):
    iter += 1
    if iter > max_iter:
        if verbose:
            print("============= Solver Exited ============")
            print(f"Number of iterariosn exceeded {max_iter}.")
            print("Increase number of max iterations by overriding default value of max_iter")
        break
    if verbose:
        print(f"============= Iteration {iter} ================")
    # Solving system
    X = np.diag(x_k)
    S = np.diag(s_k)
    e = np.ones(X.shape[0])
    mu = np.dot(x_k, s_k) / X.shape[0]
    S_inv = np.linalg.inv(S)
    D_squared = S_inv@X

    # Affine Step
    r_c = A.T@y_k + s_k - c
    r_b = A@x_k - b
    if solver != 1:
      r_xs = X@S@e
      Delta_y = np.linalg.solve(A@D_squared@A.T, - r_b - A@X@S_inv@r_c + A@S_inv@r_xs)
      Delta_s = - r_c - A.T@Delta_y
      Delta_x = - S_inv@r_xs - X@S_inv@Delta_s
      alpha_pri_aff = min(1.0, np.min(-x_k[Delta_x < 0] / Delta_x[Delta_x < 0]))
      alpha_dual_aff = min(1.0, np.min(-s_k[Delta_s < 0] / Delta_s[Delta_s < 0]))
      if verbose:
          print(f"alpha_pri_aff: {alpha_pri_aff} alpha_dual_aff:{alpha_dual_aff}")
      mu_aff = np.dot(x_k + alpha_pri_aff * Delta_x, s_k + alpha_dual_aff * Delta_s) / X.shape[0]
      sigma = (mu_aff / mu)**3


    # Final Step
    r_xs = X@S@e - sigma * mu
    if solver == 2:
      r_xs += Delta_x*Delta_s
    Delta_y = np.linalg.solve(A@D_squared@A.T, - r_b - A@X@S_inv@r_c + A@S_inv@r_xs)
    Delta_s = - r_c - A.T@Delta_y
    Delta_x = - S_inv@r_xs - X@S_inv@Delta_s

    # Update
    alpha_pri, alpha_dual = alpha, alpha
    if solver != 1:
      alpha_pri = min(1, 0.9*np.min(-x_k[Delta_x < 0] / Delta_x[Delta_x < 0]))
      alpha_dual = min(1, 0.9*np.min(-s_k[Delta_s < 0] / Delta_s[Delta_s < 0]))
    if verbose:
      print(f"alpha_pri: {alpha_pri} alpha_dual: {alpha_dual}")
      print(f"mu_aff: {mu_aff} mu: {mu}")
      print(f"sigma: {sigma}")
      print(f"Delta x: {Delta_x}")
      print(f"Delta y: {Delta_y}")
      print(f"Delta s: {Delta_s}")

    x_k, y_k, s_k = x_k + alpha_pri * Delta_x, y_k + alpha_dual* Delta_y, s_k + alpha_dual*Delta_s
    if verbose:
      print(f"x_k+1: {x_k}")
      print(f"y_k+1: {y_k}")
      print(f"s_k+1: {s_k}")
    f = c@x_k
    fs.append(f)
    xs.append(x_k)
    ss.append(s_k)
    if verbose:
      print(f"f: {f}")
  fs = np.array(fs)
  xs = np.array(xs)
  ss = np.array(ss)

  if plotting > 0:
    plt.plot(xs[:,0]*ss[:,0], xs[:,1]*ss[:,1], 'o-')
    plt.xlabel(r'$x_1s_1$')
    plt.ylabel(r'$x_2s_2$')
    plt.title(f"Complementary condition - {method}")
    if case is not None:
      plt.savefig(f"{case}- Complementary condition - {solver}.png", dpi=400,  bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.plot(range(0, iter+1), fs, 'o-')

    plt.xlabel('iteration')
    plt.ylabel(r'$f(\mathbf{x})$')
    plt.title(f"Objective function vs. iteration - {method}")
    if case is not None:
      plt.savefig(f"{case}- Objective function vs. iteration - {solver}.png", dpi=400,  bbox_inches='tight', pad_inches=0)
    plt.show()
  if plotting > 1:
    x = np.linspace(0,16,3000)
    d = np.linspace(0,16,3000)
    xx, yy = np.meshgrid(d,d)
    cond = yy<yy.max()
    i = 0
    plt.figure(figsize=(10,7))
    for r in A[:,:2]:
      y = (b[i] - r[0] * x )/r[1]
      cond = cond & (r[1]*yy<=b[i]-r[0]*xx)
      i += 1
      plt.plot(x, y)
    plt.imshow(cond.astype(int), extent=(xx.min(), xx.max(),yy.min(), yy.max()),
            origin='lower',
            cmap='Greys',
            alpha=0.3
        )

    plt.plot(xs[:,0], xs[:,1], 'o--')
    plt.xlim(0,1.5*np.max(xs[:,0]))
    plt.ylim(0,1.5*np.max(xs[:,1]))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(f"Central Path - {method}")
    if case is not None:
      plt.savefig(f"{case}- Central Path - {solver}.png", dpi=400,  bbox_inches='tight', pad_inches=0)
    plt.show()
  print(f"================== Solution Found {method} ================")
  print(f">> Finished in {iter} iterations")
  print(f">> x: {x_k}")
  print(f">> Objective Function: {f}")
  return OptResult(f, x_k)



def print_runs(A, b, c):
  start_time = time.time()
  res = linprog(c, A_eq=A, b_eq=b, method='interior-point')
  print("SciPy: --- %s seconds ---" % (time.time() - start_time))
  print(f"f : {res.fun} x: {res.x}")


  start_time = time.time()
  res = interior_point(A, b, c, verbose=False, plotting=0, solver=0)
  print(res)
  print("Adaptive: --- %s seconds ---" % (time.time() - start_time))
  print(f"f : {res.fun} x: {res.x}")


  start_time = time.time()
  res = interior_point(A, b, c, verbose=False, plotting=0, solver=1)
  print("Fixed: --- %s seconds ---" % (time.time() - start_time))
  print(f"f : {res.fun} x: {res.x}")


  start_time = time.time()
  res = interior_point(A, b, c, verbose=False, plotting=0, solver=2)
  print("Mehrotra: --- %s seconds ---" % (time.time() - start_time))
  print(f"f : {res.fun} x: {res.x}")


def save_plots(A, b, c, example_num):
  res = interior_point(A, b, c, verbose=False, plotting=2, solver=0, case=example_num)
  res = interior_point(A, b, c, verbose=False, plotting=2, solver=1, case=example_num)
  res = interior_point(A, b, c, verbose=False, plotting=2, solver=2, case=example_num)


# Case 1
A = np.array([
    [1, -1, 1, 0, 0],
    [2, -1, 0, 1, 0],
    [0, 1, 0, 0, 1]
])
b = np.array([1, 3, 5])
c = np.array([-4, -3, 0, 0, 0])
print_runs(A, b, c)
save_plots(A, b, c, 1)

# Case 2
A = np.array([
    [-8,-1,1,0,0,0],
    [-2,-1,0,1,0,0],
    [-1,-3,0,0,1,0],
    [-1,-6,0,0,0,1]
])
b = np.array([-8,-6,-6,-8])
c = np.array([3,2,0,0,0,0])
print_runs(A, b, c)
save_plots(A, b, c, 2)

# Case 3

A = np.array([
    [-1,1,1,0],
    [2,1,0,1]
    ])
b = np.array([1,2])
c = np.array([-2,-6,0,0])
print_runs(A, b, c)
save_plots(A, b, c, 3)