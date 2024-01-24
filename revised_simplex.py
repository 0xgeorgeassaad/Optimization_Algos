# pip install ortools

import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp
import time



A = np.array([
    [1, -1, 1, 0, 0],
    [2, -1, 0, 1, 0],
    [0, 1, 0, 0, 1]
    ])
b = np.array([1, 3, 5])
c = np.array([4, 3, 0, 0, 0])
B = [2, 3, 4]
N = [0, 1]


def revised_simplex(
    A: npt.NDArray,
    b: npt.NDArray,
    c: npt.NDArray,
    B: list,
    N: list,
    verbose: bool = True
) -> None:
  A_B, A_N = A[:, list(B)], A[:, list(N)]
  c_B, c_N = c[list(B)], c[list(N)]
  x_B = np.linalg.solve(A_B, b)
  iter = 0
  while True:
    iter += 1
    A_B, A_N = A[:, list(B)], A[:, list(N)]
    c_B, c_N = c[list(B)], c[list(N)]
    # choose entering variable
    v = np.linalg.solve(A_B.T, c_B)
    z_N = A_N.T @ v - c_N
    if np.all(z_N >= 0):
      if verbose:
        f = c_B.T@x_B
        print("=============== Result ==============")
        print(f"Solved in {iter} Iterations")
        print(f"x: {x_B}")
        print(f"Optimal Value is: {f}")
        print("=====================================")
      break
    t_j = z_N.argmin()
    # choose leaving variable
    Delta_x_B = np.linalg.solve(A_B, A_N[:, t_j])
    t_s = x_B / Delta_x_B
    if np.all(t_s < 0):
      print("Unbounded ")
      break
    t_i = np.where(t_s >= 0, t_s, np.inf).argmin()
    t = t_s[t_i]
    # updates
    N[t_j], B[t_i] = B[t_i], N[t_j]
    x_B = x_B - t * Delta_x_B
    x_B[t_i] = t




def simplex_google(verbose = True):
  solver = pywraplp.Solver.CreateSolver("GLOP")

  x = solver.NumVar(0, solver.infinity(), "x")
  y = solver.NumVar(0, solver.infinity(), "y")


  solver.Add(x - y <= 1)
  solver.Add(2 * x - y <= 3)
  solver.Add(2 * x - y <= 3)
  solver.Add(y <= 5)


  solver.Maximize(4 * x + 3 * y)
  status = solver.Solve()
  if verbose:
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print("Objective value =", solver.Objective().Value())
        print("x =", x.solution_value())
        print("y =", y.solution_value())
    else:
        print("The problem does not have an optimal solution.")
    print("\nAdvanced usage:")

    print("Problem solved in %f milliseconds" % solver.wall_time())
    print("Problem solved in %d iterations" % solver.iterations())



google_times = []
our_times = []
for _ in range(10):
  print(f"===== {_} Run ====")
  start_time = time.time()
  simplex_google(verbose = False)
  runtime = time.time() - start_time
  print("---Google %s seconds ---" % (runtime))
  google_times.append(runtime)

  start_time = time.time()
  revised_simplex(A[:], b[:], c[:], B[:], N[:], verbose= False)
  runtime = time.time() - start_time
  print("---OUR    %s seconds ---" % (runtime))
  our_times.append(runtime)

print("---Our MEAN %s seconds ---" % (sum(our_times) / 10))
print("---Google MEAN %s seconds ---" % (sum(google_times) / 10))