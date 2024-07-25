import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_interior_point_newton import par_interior_point_optimal_control
from noc.differential_dynamic_programming import interior_point_ddp
from noc.seq_interior_point_newton import seq_log_barrier
from noc.utils import discretize_dynamics
from jax import lax, debug
from noc.utils import wrap_angle, euler
import time
import pandas as pd
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")


def constraints(state, control):
    control_ub = 50.0
    control_lb = -50.0

    c0 = control[0] - control_ub
    c1 = -control[0] + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray, bp) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier


def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def cartpole(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )



Ts = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
N = [20, 40, 80, 100, 200, 400, 800, 1000]
ddp_time_means = []
ddp_time_medians = []
par_time_means = []
par_time_medians = []

for sampling_period, horizon in zip(Ts, N):
    ddp_time_array = []
    par_time_array = []
    downsampling = 1
    dynamics = euler(cartpole, sampling_period)

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
    _jitted_Newton = jax.jit(annon_par_Newton)
    _jitted_ddp = jax.jit(annon_ddp)

    _, _ = _jitted_Newton(u, x0)
    _, _ = _jitted_ddp(u, x0)
    for i in range(10):
        print(i)

        start = time.time()
        u_N, it_N = _jitted_Newton(u, x0)
        jax.block_until_ready(u_N)
        end = time.time()
        N_time = end - start
        print("par finished")

        start = time.time()
        u_ddp, it_ddp = _jitted_ddp(u, x0)
        jax.block_until_ready(u_ddp)
        end = time.time()
        ddp_time = end - start
        print("seq finished")

        ddp_time_array.append(ddp_time)
        par_time_array.append(N_time)

    ddp_time_means.append(jnp.mean(jnp.array(ddp_time_array)))
    ddp_time_medians.append(jnp.median(jnp.array(ddp_time_array)))
    par_time_means.append(jnp.mean(jnp.array(par_time_array)))
    par_time_medians.append(jnp.median(jnp.array(par_time_array)))

seq_time_means_arr = jnp.array(ddp_time_means)
seq_time_medians_arr = jnp.array(ddp_time_medians)
par_time_means_arr = jnp.array(par_time_means)
par_time_medians_arr = jnp.array(par_time_medians)

df_means_ddp = pd.DataFrame(seq_time_means_arr)
df_median_ddp = pd.DataFrame(seq_time_medians_arr)
df_mean_par = pd.DataFrame(par_time_means_arr)
df_median_par = pd.DataFrame(par_time_medians_arr)

df_means_ddp.to_csv("cartpole_ip_means_ddp.csv")
df_median_ddp.to_csv("cartpole_ip_medians_ddp.csv")
df_mean_par.to_csv("cartpole_ip_means_par.csv")
df_median_par.to_csv("cartpole_ip_medians_par.csv")