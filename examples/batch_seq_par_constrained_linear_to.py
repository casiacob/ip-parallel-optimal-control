import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
from noc.seq_log_barrier_optimal_control import seq_log_barrier
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics, rollout, get_QP_problem
from jax import jacrev
from jaxopt import BoxOSQP
import jax
import time

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

config.update("jax_platform_name", "cuda")


def ode(state: jnp.ndarray, control: jnp.ndarray):
    Ac = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = jnp.array([[0.0], [1.0]])
    return Ac @ state + Bc @ control

def constraints(state: jnp.ndarray, control: jnp.ndarray):
    g0 = control - 2.5
    g1 = -control - 2.5
    return jnp.hstack((g0, g1))


def stage_cost(state: jnp.ndarray, control: jnp.ndarray, bp: float):
    X = jnp.diag(jnp.array([1e2, 1e0]))
    U = 1e-1 * jnp.eye(control.shape[0])
    c = 0.5 * state.T @ X @ state + 0.5 * control.T @ U @ control
    log_barrier = jnp.sum(jnp.log(-constraints(state, control)))
    return c - bp * log_barrier


def final_cost(state: jnp.ndarray):
    P = jnp.diag(jnp.array([1e2, 1e0]))
    return 0.5 * state.T @ P @ state


def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(stage_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)

step = 0.01
horizon = 600
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)
x0 = jnp.array([2.0, 1.0])
u = jnp.zeros((horizon, 1))
linear_problem = OCP(dynamics, constraints, stage_cost, final_cost, total_cost)


par_sol = lambda u, x0: par_log_barrier(linear_problem, u, x0)
seq_sol = lambda u, x0: seq_log_barrier(linear_problem, u, x0)

_jitted_par_log_barrier = jax.jit(par_sol)
_jitted_seq_log_barrier = jax.jit(seq_sol)


x_par, u_par = _jitted_par_log_barrier( u, x0)
x_seq, u_seq = _jitted_seq_log_barrier( u, x0)

start = time.time()
x_par, u_par = _jitted_par_log_barrier( u, x0)
x_par.block_until_ready()
end = time.time()
par_time = end - start

start = time.time()
x_seq, u_seq = _jitted_seq_log_barrier( u, x0)
x_seq.block_until_ready()
end = time.time()
seq_time = end - start

# define batch problem
N = horizon
QN = jnp.diag(jnp.array([1e2, 1e0]))
Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(1)
x0 = jnp.array([2.0, 1.0])
Ad = jacrev(dynamics, 0)(jnp.array([0., 0.]), jnp.array([0.]))
Bd = jacrev(dynamics, 1)(jnp.array([0., 0.]), jnp.array([0.]))
umin = -2.5
umax = 2.5

H, g, A, upper = get_QP_problem(Ad, Bd, QN, Q, R, N, jnp.eye(1), umax, umin)
lower = -jnp.inf * jnp.ones(upper.shape[0])
qp = BoxOSQP(jit=True)
sol = qp.run(params_obj=(H, g@x0), params_eq=A, params_ineq=(lower, upper))

start = time.time()
batch_sol = qp.run(params_obj=(H, g@x0), params_eq=A, params_ineq=(lower, upper))
jax.block_until_ready(batch_sol.params.primal[0])
end = time.time()
batch_time = end-start
u_batch = batch_sol.params.primal[0]

print('par time           : ', par_time)
print('seq time           : ', seq_time)
print('batch time         : ', batch_time)
print('|x1_par - x1_seq|  : ', jnp.max(jnp.abs(x_par[:, 0] - x_seq[:, 0])))
print('|x2_par - x2_seq|  : ', jnp.max(jnp.abs(x_par[:, 1] - x_seq[:, 1])))
print('|u_par  - u_seq|   : ', jnp.max(jnp.abs(u_par-u_seq)))
print('|u_par  - u_batch| : ', jnp.max(jnp.abs(u_par-u_batch.reshape(-1, 1))))
print('|u_seq  - u_batch| : ', jnp.max(jnp.abs(u_par-u_batch.reshape(-1, 1))))