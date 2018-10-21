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
  print("WARNING: qpoases not found; recommended qpOASESSolver unavailable.")
  Solver = qp_solvers.ScipySolver
else:
  print("One of scipy or qpoases required (qpoases recommended).")
  import sys
  sys.exit(1)

def constrained_lqr(A, B, Q, R, N, max_abs_u):
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

  default_x_ref = np.zeros(calB.shape[0])
  default_u_ref = np.zeros(calB.shape[1])

  BtQ = calB.T.dot(calQ)

  def get_h(x, x_ref, u_ref):
    return BtQ.dot(vecc_gen.dot(x) - x_ref) - calR.dot(u_ref)

  def controller_inv(x, x_ref=default_x_ref, u_ref=default_u_ref):
    return calHinv[:B.shape[1], :].dot(-get_h(x, x_ref, u_ref))

  solver = Solver(N, max_abs_u)

  def controller_opt(x, x_ref=default_x_ref, u_ref=default_u_ref):
    return solver.solve(calH, get_h(x, x_ref, u_ref))[:B.shape[1]]

  return controller_inv, controller_opt

def get_clipped_controller(controller, abs_u):
  return lambda x, *args, **kwargs: np.clip(controller(x, *args, **kwargs), -abs_u, abs_u)

if __name__ == "__main__":
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument("--ref", required=False, default=False, action="store_true")
  args = parser.parse_args()

  w_pos = 1e5
  #w_vel = 3e3
  w_vel = 1e-1

  max_abs_u = 8

  Q = np.array(((w_pos, 0.0), (0.0, w_vel)))
  R = np.array(((1.0,),))

  dt = 0.01
  A = np.array(((1, dt), (0, 1)))
  B = np.array(((0,), (dt,)))

  N = 100 # Horizon
  t_end = 2.0

  distance = 1.0
  x_0 = np.array((distance, 0))

  no_steps = int(round(t_end / dt) + N)

  def make_ref(distance, accel, t_end, dt):
    """ Creates x_ref and u_ref from bang bang acceleration control. """

    u_ref = np.zeros(no_steps)
    accel_time = (distance / accel) ** 0.5
    accel_steps = int(round(accel_time / dt))
    u_ref[:accel_steps] = -accel
    u_ref[accel_steps : 2 * accel_steps] = accel

    vel_ref = np.cumsum(u_ref) * dt
    pos_ref = np.cumsum(vel_ref) * dt + distance
    # Ensure that the reference starts with the starting position... hm...
    pos_ref = np.hstack((np.array((distance,)), pos_ref[:-1]))

    x_ref = np.zeros(no_steps * 2)
    x_ref[0::2] = pos_ref
    x_ref[1::2] = vel_ref

    return x_ref, u_ref

  if args.ref:
    x_ref, u_ref = make_ref(distance, 1.4 * max_abs_u, t_end, dt)
  else:
    x_ref = np.zeros(2 * no_steps)
    u_ref = np.zeros(no_steps)

  lqr, lqr_qp = constrained_lqr(A, B, Q, R, N, max_abs_u)
  lqr_clip = get_clipped_controller(lqr, max_abs_u)

  controllers = [lqr, lqr_clip, lqr_qp]
  names = ["LQR", "LQR Clipped", "LQR QP"]

  ts = np.arange(0, t_end, dt)

  f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
  for controller, name in zip(controllers, names):
    xs = [x_0]
    us = []

    cost = 0

    for i, t in enumerate(ts[:-1]):
      x = xs[-1]

      def make_short_ref(ref, s):
        return ref[s * i : s * (i + N)]

      x_ref_short = make_short_ref(x_ref, 2)
      u_ref_short = make_short_ref(u_ref, 1)
      u = controller(x, x_ref=x_ref_short, u_ref=u_ref_short)

      next_x = A.dot(x) + B.dot(u)

      x_err = next_x - x_ref_short[:len(x)]
      u_err = u - u_ref_short[:len(u)]
      cost += x_err.T.dot(Q).dot(x_err) + u_err.T.dot(R).dot(u_err)

      xs.append(next_x)
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
