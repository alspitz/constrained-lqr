import numpy as np

try:
  import scipy.optimize
except ImportError:
  scipy_available = False
else:
  scipy_available = True

try:
  import qpoases
except ImportError:
  qpoases_available = False
else:
  qpoases_available = True

if scipy_available:
  class ScipySolver:
    def __init__(self, N, max_abs_u):
      self.last_u = np.zeros(N)
      self.bounds = scipy.optimize.Bounds([-max_abs_u] * N, [max_abs_u] * N)

    def solve(self, H, h):
      def to_min(u):
        return 0.5 * u.T.dot(H).dot(u) + h.T.dot(u)

      def jac(u):
        return u.T.dot(H) + h.T

      def hess(u):
        return H

      res = scipy.optimize.minimize(to_min, self.last_u, method='trust-constr',
                                    jac=jac, hess=hess, bounds=self.bounds)
      self.last_u = res.x
      return res.x

if qpoases_available:
  class qpOASESSolver:
    def __init__(self, N, max_abs_u):
      self.ub = np.array([max_abs_u] * N, dtype=np.float)
      self.lb = -self.ub
      self.N = N

      # Use "Simply Bounded" quadratic program.
      self.problem = qpoases.PyQProblemB(self.N)
      options = qpoases.PyOptions()
      options.printLevel = qpoases.PyPrintLevel.NONE
      self.problem.setOptions(options)

    def solve(self, H, h):
      self.nWSR = np.array([1000])
      ret = self.problem.init(H, h, self.lb, self.ub, self.nWSR)
      if ret:
        print("qpOASES QP solve failed! (%d)" % ret)

      x = np.zeros(self.N)
      self.problem.getPrimalSolution(x)
      return x
