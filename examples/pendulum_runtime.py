import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_interior_point_newton import par_interior_point_optimal_control
from noc.differential_dynamic_programming import interior_point_ddp
from noc.seq_interior_point_newton import seq_interior_point_optimal_control
from noc.utils import euler
from noc.utils import wrap_angle
import time
import pandas as pd

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")


def constraints(state, control):
    # control_ub = jnp.finfo(jnp.float64).max
    # control_lb = jnp.finfo(jnp.float64).min
    control_ub = 5.0
    control_lb = -5.0

    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state):
    goal_state = jnp.array((jnp.pi, 0.0))
    final_state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action, bp):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier


def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )

Ts = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
N = [20, 40, 80, 100, 200, 400, 800, 1000]*3
ddp_time_means = []
ddp_time_medians = []
par_time_means = []
par_time_medians = []
seq_time_means = []
seq_time_medians = []

for sampling_period, horizon in zip(Ts, N):
    ddp_time_array = []
    par_time_array = []
    seq_time_array = []
    downsampling = 1
    dynamics = euler(pendulum, sampling_period)

    x0 = jnp.array([wrap_angle(0.1), -0.1])
    key = jax.random.PRNGKey(1)
    u = 0.1 * jax.random.normal(key, shape=(horizon, 1))
    nonlinear_problem = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)

    annon_par_Newton = lambda init_u, init_x0: par_interior_point_optimal_control(
        nonlinear_problem, init_u, init_x0
    )
    annon_ddp = lambda init_u, init_x0: interior_point_ddp(
        nonlinear_problem, init_u, init_x0
    )
    annon_seq_Newton = lambda init_u, init_x0: seq_interior_point_optimal_control(
        nonlinear_problem, init_u, init_x0
    )
    _jitted_Newton = jax.jit(annon_par_Newton)
    _jitted_ddp = jax.jit(annon_ddp)
    _jitted_seq = jax.jit(annon_seq_Newton)

    _, _ = _jitted_Newton(u, x0)
    _, _ = _jitted_ddp(u, x0)
    _, _ = _jitted_seq(u, x0)
    for i in range(10):
        start = time.time()
        u_N, it_N = _jitted_Newton(u, x0)
        jax.block_until_ready(u_N)
        end = time.time()
        N_time = end - start

        start = time.time()
        u_ddp, it_ddp = _jitted_ddp(u, x0)
        jax.block_until_ready(u_ddp)
        end = time.time()
        ddp_time = end - start

        start = time.time()
        u_seq, it_seq = _jitted_seq(u, x0)
        jax.block_until_ready(u_seq)
        end = time.time()
        seq_time = end - start

        ddp_time_array.append(ddp_time)
        par_time_array.append(N_time)
        seq_time_array.append(seq_time)

    ddp_time_means.append(jnp.mean(jnp.array(ddp_time_array)))
    ddp_time_medians.append(jnp.median(jnp.array(ddp_time_array)))
    par_time_means.append(jnp.mean(jnp.array(par_time_array)))
    par_time_medians.append(jnp.median(jnp.array(par_time_array)))
    seq_time_means.append(jnp.mean(jnp.array(seq_time_array)))
    seq_time_medians.append(jnp.median(jnp.array(seq_time_array)))

ddp_time_means_arr = jnp.array(ddp_time_means)
ddp_time_medians_arr = jnp.array(ddp_time_medians)
par_time_means_arr = jnp.array(par_time_means)
par_time_medians_arr = jnp.array(par_time_medians)
seq_time_means_arr = jnp.array(seq_time_means)
seq_time_medians_arr = jnp.array(seq_time_medians)

df_means_ddp = pd.DataFrame(ddp_time_means_arr)
df_median_ddp = pd.DataFrame(ddp_time_medians_arr)
df_mean_par = pd.DataFrame(par_time_means_arr)
df_median_par = pd.DataFrame(par_time_medians_arr)
df_mean_seq = pd.DataFrame(seq_time_means_arr)
df_median_seq = pd.DataFrame(seq_time_medians_arr)


df_means_ddp.to_csv("pendulum_ip_means_ddp.csv")
df_median_ddp.to_csv("pendulum_ip_medians_ddp.csv")
df_mean_par.to_csv("pendulum_ip_means_par.csv")
df_median_par.to_csv("pendulum_ip_medians_par.csv")
df_mean_seq.to_csv("pendulum_ip_means_seq.csv")
df_median_seq.to_csv("pendulum_ip_medians_seq.csv")

