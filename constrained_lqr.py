"""
  Consider a 1D point mass:
    dx = v
    dv = u

  minimize w_pos * x ** 2 + w_pos * v ** 2 + u ** 2
  with u in [-max_abs_u, max_abs_u]
"""

import numpy as np

import qp_solvers

if qp_solvers.qpoases_available:
  Solver = qp_solvers.qpOASESSolver
elif qp_solvers.scipy_available:
  print("WARNING: qpoases not found; reocmmended qpOASESSolver unavailable.")
  Solver = qp_solvers.ScipySolver
else:
  print("One of scipy or qpoases required (qpoases recommended).")
  import sys
  sys.exit(1)

w_pos = 1e5
#w_vel = 3e3
w_vel = 1e-1

max_abs_u = 8

Q = np.array(((w_pos, 0.0), (0.0, w_vel)))
R = np.array(((1.0,),))

dt = 0.01
A = np.array(((1, dt), (0, 1)))
B = np.array(((0,), (dt,)))

x_0 = np.array((1, 0))

N = 100 # Horizon
t_end = 2.0

def constrained_lqr(A, B, Q, R):
  """
           B          0          ...   0
           AB         B          ...   0
           .          .       .        .
    calB = .          .          .     .
           .          .             .  .
           A^(N-1)B   A^(N-2)B   ...   B
  """
  calB = np.zeros((N * A.shape[0], N * B.shape[1]))
  vecc_gen = np.zeros((A.shape[0] * N, A.shape[0]))

  Ap = np.eye(A.shape[0])
  for i in range(N):
    for j in range(N - i):
      calB[A.shape[0] * (i + j) : A.shape[0] * (i + j + 1),
           B.shape[1] * j       : B.shape[1] * (j + 1)] = Ap.dot(B)
    Ap = A.dot(Ap)

    vecc_gen[A.shape[0] * i : A.shape[0] * (i + 1), :] = Ap

  # Build block diagonal cost matrices.
  calQ = np.kron(np.eye(N), Q)
  calR = np.kron(np.eye(N), R)
  calH = calB.T.dot(calQ).dot(calB) + calR
  calHinv = np.linalg.inv(calH)

  K = calHinv[0:B.shape[1], :].dot(calB.T.dot(calQ).dot(vecc_gen))
  print("LQR gain", K)

  def controller_inv(x):
    return -K.dot(x)

  solver = Solver(N, max_abs_u)

  def controller_opt(x):
    vech = calB.T.dot(calQ).dot(vecc_gen.dot(x))
    vecu = solver.solve(calH, vech)
    return vecu[0:B.shape[1]]

  return controller_inv, controller_opt

def get_gain_controller(K):
  return lambda x: -K.dot(x)

def get_clipped_controller(controller, abs_u):
  return lambda x: np.clip(controller(x), -abs_u, abs_u)

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  lqr, lqr_qp = constrained_lqr(A, B, Q, R)
  lqr_clip = get_clipped_controller(lqr, max_abs_u)

  controllers = [lqr, lqr_clip, lqr_qp]
  names = ["LQR", "LQR Clipped", "LQR QP"]

  ts = np.arange(0, t_end, dt)

  f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
  for controller, name in zip(controllers, names):
    xs = [x_0]
    us = []

    cost = 0

    for t in ts[:-1]:
      x = xs[-1]
      u = controller(x)

      cost += x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)

      xs.append(A.dot(x) + B.dot(u))
      us.append(u)

    xs = np.array(xs)
    us = np.array(us)

    print(cost, "for", name)

    ax1.plot(ts, xs[:, 0], label=name)
    ax2.plot(ts, xs[:, 1], label=name)
    ax3.plot(ts[1:], us[:, 0], label=name)

  ax1.set_title("Position")
  ax2.set_title("Velocity")
  ax3.set_title("Acceleration (Input)")
  ax1.set_ylabel('Pos.')
  ax2.set_ylabel('Vel.')
  ax3.set_ylabel('Acc.')
  ax3.set_ylim((-1.5 * max_abs_u, 1.5 * max_abs_u))
  ax3.set_xlabel('Time (s)')
  ax1.legend(loc='upper right', ncol=len(controllers), fancybox=True, shadow=True)
  plt.tight_layout()
  plt.show()
