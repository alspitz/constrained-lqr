# LQR with Input Constraints

Simulates a 1D point mass controlled using LQR with input constraints. Solving the constrained Quadratic Program (LQR QP) can in many cases do better than simply thresholding the output of an unconstrained LQR controller.

![alt text](sample.png "Simulation Data")

## Usage

Run [constrained_lqr.py](constrained_lqr.py). Pass `--ref` to follow a bang bang reference that violates input constraints.

Change `w_pos`, `w_vel`, `max_abs_u`, and `N` to experiment with different cost functions, input constraints, and horizon lengths.

## Dependencies

* numpy
* matplotlib
* one of scipy or [qpoases](https://projects.coin-or.org/qpOASES/)' Python interface to solve the quadratic program. qpoases runs much faster and is recommended.

## Issues

If you get an undefined symbol error when using qpoases, see <https://projects.coin-or.org/qpOASES/ticket/70>.
