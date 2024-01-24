import numpy as np
from scipy import optimize
from collections.abc import Sequence
import torch
import time


# 1D Minimization

class OptimizeResult(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def _section_search(f, brack, ratios, tol=np.sqrt(np.finfo(float).eps), verbose=False):
  assert len(brack) ==2, f"brack length should be 2, got {len(brack)}"
  assert isinstance(ratios, Sequence)
  funcalls, nit = 0, 0
  x_l, x_u = brack
  d = ratios[0] * (x_u - x_l)
  x_1, x_2 = x_l + d, x_u - d
  f_1, f_2 = f(x_1), f(x_2)
  funcalls += 2
  if verbose:
    print(f"x_l: {x_l}, x_2: {x_2}, f_2: {f_2}, x_1: {x_1}, f_1: {f_1}, x_u: {x_u}")
  for i in range(1,len(ratios)):
    if np.abs(x_u - x_l) <= tol * (np.abs(x_1) + np.abs(x_2)):
      break
    if f_1 > f_2:
      x_u = x_1
      x_1, f_1 = x_2, f_2
      d = ratios[i] * (x_u - x_l)
      x_2 = x_u - d
      f_2 = f(x_2)
    else:
      x_l = x_2
      x_2, f_2 = x_1, f_1
      d = ratios[i] * (x_u - x_l)
      x_1 = x_l + d
      f_1 = f(x_1)
    if verbose:
      print(f"x_l: {x_l}, x_2: {x_2}, f_2: {f_2}, x_1: {x_1}, f_1: {f_1}, x_u: {x_u}")
    funcalls += 1
    nit += 1

  if (f_1 < f_2):
      xmin = x_1
      fval = f_1
  else:
      xmin = x_2
      fval = f_2

  success = nit < ratios.maxiter() and not (np.isnan(fval) or np.isnan(xmin))

  if success:
    message = ("\nOptimization terminated successfully;\n"
                "The returned value satisfies the termination criteria\n"
                f"(using xtol = {tol} )")
  else:
    message = "\nMaximum number of iterations Reached"

  return OptimizeResult(fun=fval, nfev=funcalls, x=xmin, nit=nit,
                        success=success, message=message)


def golden(f, brack, max_iter=10, tol=np.sqrt(np.finfo(float).eps), verbose=False):
  assert len(brack) ==2, f"brack length should be 2, got{len(brack)}"
  class GoldenSection(Sequence):
    def __init__(self, maxiter):
      self._gR = 0.61803399
      self._maxiter = maxiter + 1
    def __getitem__(self, index):
      return self._gR
    def __len__(self):
      return self._maxiter
    def maxiter(self):
      return self._maxiter - 1

  return _section_search(f, brack, GoldenSection(max_iter), tol=tol, verbose=verbose)


def fibonacci(f, brack, N, tol=np.sqrt(np.finfo(float).eps), verbose=False):
  assert len(brack) ==2, f"brack length should be 2, got{len(brack)}"
  class FibonacciSection(Sequence):
    def __init__(self, maxiter):
      fib_nos = [1, 1]
      for i in range(2,maxiter):
          fib_nos.append(fib_nos[i-1]+fib_nos[i-2])
      self._ratios = []
      for i in range(len(fib_nos)-1,1,-1):
        self._ratios.append(fib_nos[i-1]/fib_nos[i])
    def __getitem__(self, index):
      return self._ratios[index]
    def __len__(self):
      return len(self._ratios)
    def maxiter(self):
      return len(self._ratios) + 1

  return _section_search(f, brack, FibonacciSection(N), tol=tol, verbose=verbose)


def quadratic(f, t_0, tol=1e-10):
  A, B, C = 0, t_0, 2*t_0
  f_A, f_B, f_C = f(A), f(B), f(C)
  funcalls = 3
  if f(t_0) > f_A:
    B, f_B = t_0/2, f(t_0/2)
  else:
    B, f_B = t_0, f(t_0)
    C, f_C = 2*t_0, f(2*t_0)
    while f_C < f_B:
      B, f_B = C, f_C
      t_0 = 2*t_0
      C, f_C = 2*t_0, f(2*t_0)
  def refit(f, A, B, C, f_A, f_B, f_C):
    a =(f_A*B*C*(C - B) + f_B*C*A*(A - C) + f_C*A*B*(B - A))/ ((A - B)*(B - C)*(C - A))
    b = (f_A*(B**2 - C**2) + f_B*(C**2 - A**2) + f_C*(A**2 - B**2))/((A - B)*(B - C)*(C - A))
    c = - (f_A*(B - C) + f_B*(C - A) + f_C*(A - B))/((A - B)*(B - C)*(C - A))
    lambda_opt = (f_A*(B**2 - C**2) + f_B*(C**2 - A**2) + f_C*(A**2 - B**2))/(2*(f_A*(B - C) + f_B*(C - A) + f_C*(A - B)))
    return a, b, c, lambda_opt
  a, b, c, lambda_opt = refit(f, A, B, C,f_A, f_B, f_C)
  h_lambda_opt = a + b * lambda_opt + c * lambda_opt**2
  f_lambda_opt = f(lambda_opt)
  funcalls += 1
  it = 0
  while np.abs((h_lambda_opt-f_lambda_opt)/(f_lambda_opt)) > tol:
    all_pts = np.array([A, B, lambda_opt, C])
    all_funs = np.array([f_A, f_B, f_lambda_opt, f_C])
    idx = np.argpartition(all_funs, 3)
    A, B, C = all_pts[idx][:3]
    f_A, f_B, f_C = all_funs[idx][:3]
    a, b, c, lambda_opt = refit(f, A, B, C, f_A, f_B, f_C)
    h_lambda_opt = a + b * lambda_opt + c * lambda_opt**2
    f_lambda_opt = f(lambda_opt)
    funcalls += 1
    it += 1
  return OptimizeResult(fun=f_lambda_opt, nfev=funcalls, x=lambda_opt, nit=it)

"""
    The following code is extremely elegant (IMHO). 
    What we do is instead of using a `grad` callable the same way SciPy does it, 
        we use the beautiful automatic differentiation with some caveats.
    1. The function `f` being passed must be torch compitable or in other words it must use 
        function from the torch api such as `torch.sin()` instead of `np.sin()`. For multiplication and power, nothing needs to be done because all these operators are already defined for the tensors.
    2. In the implementation, special attention must be paid to detaching tensors from the computational graph 
        before calculating the gradient, or otherwise the gradient will be wrong
    3. This implementation provides one major advantage and 
        it's that there is no need to explicitly pass a callable for the gradient which is very handy.
"""

def cubic(f, A, B, tol=1e-4, verbose=False):
  A = torch.tensor(A, dtype=torch.float64, requires_grad = True)
  B = torch.tensor(B, dtype=torch.float64, requires_grad = True)
  def refit(f, A, B):
    f_A, f_B = f(A), f(B)
    f_A.backward()
    f_B.backward()
    f_A_prime, f_B_prime = A.grad, B.grad
    Z = 3*(f_A - f_B)/(B -A) + f_A_prime + f_B_prime
    b = (B**2*f_A_prime + A**2*f_B_prime +2*A*B*Z )/(A-B)**2
    c = -((A+B)*Z + B*f_A_prime + A*f_B_prime)/(A-B)**2
    d = (2*Z + f_A_prime + f_B_prime)/(3*(A-B)**2)
    a = f_A - b*A - c*A**2 - d*A**3
    assert not torch.isnan(b).any()
    assert not torch.isnan(c).any()
    assert not torch.isnan(d).any()
    #print(f_A_prime, f_B_prime)
    assert c**2-3*b*d >= 0
    lambda_1 = (-c+(c**2-3*b*d)**(1/2))/(3*d)
    lambda_2 = (-c-(c**2-3*b*d)**(1/2))/(3*d)
    assert not torch.isnan(lambda_1).any()
    assert not torch.isnan(lambda_2).any()
    if lambda_1 >= A and lambda_1 <= B:
      lambda_opt = lambda_1
    else:
      lambda_opt = lambda_2
    return lambda_opt.clone().detach().requires_grad_(True)
  lambda_opt = refit(f, A, B)
  f_lambda_opt = f(lambda_opt)
  f_lambda_opt.backward()
  it = 0
  funcalls = 2
  if verbose:
    print(f"lambda_opt: {lambda_opt} grad: {lambda_opt.grad}")
  while torch.abs(lambda_opt.grad) > tol:
    if lambda_opt.grad < 0:
      A, B = lambda_opt.clone().detach().requires_grad_(True), B.clone().detach().requires_grad_(True)
    else:
      A, B = A.clone().detach().requires_grad_(True), lambda_opt.clone().detach().requires_grad_(True)
    lambda_opt = refit(f, A, B)
    f_lambda_opt = f(lambda_opt)
    f_lambda_opt.backward()
    it += 1
    funcalls += 2
    if verbose:
      print(f"lambda_opt: {lambda_opt} grad: {lambda_opt.grad}")
  if verbose:
    print("Finished, Found Opt Lambda")
  return OptimizeResult(
        fun=f_lambda_opt.detach().numpy(),
        nfev=funcalls,
        x=lambda_opt.detach().numpy(),
        nit=it
  )


def gradient_descent(f, x_0, tol=1e-2, verbose=False):
  lam = torch.tensor(x_0, dtype=torch.float64, requires_grad = True)
  optimizer = torch.optim.SGD([lam], lr=1e-5, momentum=0.7)
  optimizer.zero_grad()
  loss = f(lam)
  loss.backward()
  funcalls = 1
  it = 0
  while torch.abs(lam.grad) > tol:
    if verbose:
      print(f"lam:{lam} grad: {lam.grad} loss: {loss}")
    optimizer.step()
    it += 1
    funcalls += 1
    optimizer.zero_grad()
    loss = f(lam)
    loss.backward()
  lam = lam.detach().numpy()
  f_val = f(lam)
  return OptimizeResult(
        fun=f_val,
        nfev=funcalls,
        x=lam,
        nit=it
  )

def f(x):
  y = 4 - 16 * x + 6416 * x**2 - 25600 * x**3 + 25600 * x**4
  return y
bracket = (0.0, 0.1)
max_iter = 15
N = 14
res_golden = golden(f, bracket, max_iter)
res_fib = fibonacci(f, bracket, N, verbose=False)
res_quad = quadratic(f, 0.001, 1e-10)
res_cubic = cubic(f,0,0.1)
res_grad_descent = gradient_descent(f, 0.1)
x, fun, nfev = res_golden.x, res_golden.fun, res_golden.nfev
x2, fun2, nfev2 = res_fib.x, res_fib.fun, res_fib.nfev
x3, fun3, nfev3 = optimize.golden(f, brack=bracket, maxiter=max_iter, full_output=True)
x4, fun4, nfev4 = res_quad.x, res_quad.fun, res_quad.nfev
x5, fun5, nfev5 = res_cubic.x, res_cubic.fun, res_cubic.nfev
x6, fun6, nfev6 = res_grad_descent.x, res_grad_descent.fun, res_grad_descent.nfev
print("{:25s} {}, {}, {}".format("Golden Section: ", x, fun, nfev))
print("{:25s} {}, {}, {}".format("Fibonacci: ", x2, fun2, nfev2))
print("{:25s} {}, {}, {}".format("Golden Section (SciPy): ", x3, fun3, nfev3))
print("{:25s} {}, {}, {}".format("Quadratic: ", x4, fun4, nfev4))
print("{:25s} {}, {}, {}".format("Cubic: ", x5, fun5, nfev5))
print("{:25s} {}, {}, {}".format("Gradient descent: ", x6, fun6, nfev6))


# Unconstrainted Optimization

def marquardt(f, grad, hess, x_1, tol=1e-6):
  c_1, c_2 = 0.25, 2
  x_i = x_1
  alpha_i = 1e4
  I = np.identity(len(x_1))
  it = 0
  while np.linalg.norm(grad(x_i)) > tol:
    f_i = f(x_i)
    x_i = x_i - np.linalg.inv(hess(x_i)+alpha_i*I)@grad(x_i)
    if f(x_i) < f_i:
      alpha_i *= c_1
    else:
      alpha_i *= c_2
    it += 1
  return it, x_i, f(x_i)

def fletcher_reeves_CG(f, grad, hess, x_1, tol=1e-6, verbose=False):
  n = len(x_1)
  s_1 = - grad(x_1)
  lam = (np.inner(s_1,s_1))/(s_1.T@hess(x_1)@s_1)
  x_2 = x_1 + lam * s_1
  x_i, grad_i, s_i = x_2, grad(x_1), s_1
  if verbose:
    print(f"x: {x_1}, grad: {s_1}")
  it = 0
  while np.linalg.norm(grad(x_i)) > tol:
    grad_i, grad_i_1 = grad(x_i), grad_i
    s_i = - grad_i + (np.inner(grad_i,grad_i))/(np.inner(grad_i_1,grad_i_1)) * s_i
    if it % n == 0:
      s_i = - grad_i
    lam = (np.inner(grad_i,grad_i))/(s_i.T@hess(x_i)@s_i)
    if verbose:
      print(f"x: {x_i}, grad: {grad_i}, s_i: {s_i} lam:{lam}")
    x_i = x_i + lam * s_i
    it += 1
    if it > 250:
      break
  if verbose:
    print(f"it: {it}, grad: {np.linalg.norm(grad(x_i))}")
  return it, x_i, f(x_i)


def quasi_newton(f, grad, x_1, tol=1e-2, verbose=False):
  x_i, B_i = x_1, np.identity(len(x_1))
  s_i = -B_i@grad(x_i)
  def ff(x_i, s_i):
    x_i = torch.from_numpy(x_i)
    s_i = torch.from_numpy(s_i)
    return lambda lam: f(x_i+lam*s_i)
  def find_AB(x_i, s_i):
    A = 0
    while np.dot(s_i, grad(x_i + A * s_i)) > 0:
      if np.dot(s_i, grad(x_i + (A-0.1) * s_i)) < np.dot(s_i, grad(x_i + (A+0.1) * s_i)):
        A -= 0.1
      else:
        A += 0.1
    B = 2*A
    while np.dot(s_i, grad(x_i + B * s_i)) < 0:
      if np.dot(s_i, grad(x_i + (B-0.1) * s_i)) > np.dot(s_i, grad(x_i + (B+0.1) * s_i)):
        B -= 0.1
      else:
        B += 0.1
    return A, B
  A, B = find_AB(x_i, s_i)
  res = cubic(ff(x_i, s_i), A, B)
  lam = res.x
  x_i, x_i_1 = x_i + lam * s_i, x_i
  it = 0
  while np.linalg.norm(grad(x_i)) > tol:
    grad_i, grad_i_1 = grad(x_i), grad(x_i_1)
    if verbose:
      print(f"x_i: {x_i} grad: {grad_i}")
    g_i = grad_i - grad_i_1
    M_i = lam * np.dot(s_i, s_i)/np.dot(s_i, g_i)
    N_i = - np.dot(B_i@g_i, B_i@g_i)/(g_i.T@B_i@g_i)
    B_i = B_i + M_i + N_i
    if (it+1) % 10 == 0:
      B_i = np.identity(len(x_1))
    s_i = -B_i@grad_i
    A, B = find_AB(x_i, s_i)
    res = cubic(ff(x_i, s_i), A, B)
    lam = res.x
    x_i, x_i_1 = x_i + lam * s_i, x_i
    it += 1
  return it, x_i, f(x_i)


x = np.array([0,0])
f = lambda x: x[0] - x[1] + 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2
f_grad = lambda x: np.array([1+4*x[0]+2*x[1], -1+2*x[0]+2*x[1]])
f_hess = lambda x: np.array([[4,2],[2,2]])
fletcher_reeves_CG(f, f_grad, f_hess, x)



x = np.array([-1.2,1])
rosenbrock = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
rosenbrock_grad = lambda x : np.array([-2*(1 - x[0]) - 400*x[0] *(x[1] - x[0]**2), 200*(x[1] - x[0]**2)])
rosenbrock_hess = lambda x : np.array([[800*x[0]**2 - 400*(x[1] - x[0]**2) + 2 , -400*x[0]],[-400*x[0] ,200]])
x = np.array([-1.2,1])
# Check that our definitions are correct
assert rosenbrock(x) == optimize.rosen(x)
assert (rosenbrock_grad(x) == optimize.rosen_der(x)).all()
assert (rosenbrock_hess(x) == optimize.rosen_hess(x)).all()


x = np.array([3,-1.0,0.0,1.0])
powell = lambda x: (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4
powell_grad = lambda x: np.array([40*(x[0] - x[3])**3 + 2*(x[0] + 10*x[1]),
                                    4*(x[1] - 2*x[2])**3 + 20*(x[0] + 10*x[1]),
                                    10*(x[2] - x[3]) - 8*(x[1] - 2*x[2])**3,
                                    -40*(x[0] - x[3])**3 - 10*(x[2] - x[3])])
powell_hess = lambda x: np.array([[120*(x[0] - x[3])**2 + 2 , 20 , 0 , -120*(x[0] - x[3])**2],
                                  [20 , 12*(x[1] - 2*x[2])**2 + 200 , -24*(x[1] - 2*x[2])**2 , 0],
                                  [0 , -24*(x[1] - 2*x[2])**2 , 48*(x[1] - 2*x[2])**2 + 10 , -10],
                                  [-120*(x[0] - x[3])**2 , 0 , -10 , 120*(x[0] - x[3])**2 + 10]])

print(powell(x))
print(powell_grad(x))
print(powell_hess(x))

x = np.array([3,-1.0,0.0,1.0])
it_1, x_1, f_1 = marquardt(powell, powell_grad, powell_hess, x)
it_2, x_2, f_2 = fletcher_reeves_CG(powell, powell_grad, powell_hess, x, verbose=False)
it_3, x_3, f_3 = quasi_newton(powell, powell_grad, x, verbose=False)
print(it_1, x_1, f_1, powell(x_1), np.linalg.norm(powell_grad(x_1)))
print(it_2, x_2, f_2, powell(x_2), np.linalg.norm(powell_grad(x_2)))
print(it_3, x_3, f_3, powell(x_3), np.linalg.norm(powell_grad(x_3)))


# Comparison

def print_runtimes(f, grad, hess, x):
  start = time.time()
  it_marquardt, x_marquardt, f_marquardt = marquardt(f, grad, hess, x)
  time_marquardt = time.time() - start

  start = time.time()
  it_cg, x_cg, f_cg = fletcher_reeves_CG(f, grad, hess, x, verbose=False)
  time_cg = time.time() - start

  start = time.time()
  it_quasi, x_quasi, f_quasi = quasi_newton(f, grad, x, verbose=False)
  time_quasi = time.time() - start
  print("{:20s} nit: {}, x_min: {}, f_min: {}, grad: {}, time: {}".format(
      "Marquardt: ", it_marquardt, x_marquardt, f_marquardt, np.linalg.norm(grad(x_marquardt)), time_marquardt))
  print("{:20s} nit: {}, x_min: {}, f_min: {}, grad: {}, time: {}".format(
      "Fletcher Reeves CG: ", it_cg, x_cg, f_cg, np.linalg.norm(grad(x_cg)), time_cg))
  print("{:20s} nit: {}, x_min: {}, f_min: {}, grad: {}, time: {}".format(
      "Quasi Newton: ", it_quasi, x_quasi, f_quasi, np.linalg.norm(grad(x_quasi)), time_quasi))
  

x = np.array([-1.2,1])
print_runtimes(rosenbrock, rosenbrock_grad, rosenbrock_hess, x)

x = np.array([3,-1.0,0.0,1.0])
print_runtimes(powell, powell_grad, powell_hess, x)