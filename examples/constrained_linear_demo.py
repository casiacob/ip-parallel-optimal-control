import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
from noc.seq_log_barrier_optimal_control import seq_log_barrier
from noc.utils import discretize_dynamics, rollout
import jax
import time
import pandas as pd

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

config.update("jax_platform_name", "cuda")

def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


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


#Ts = [0.1, 0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
#N = [60, 120, 240, 480, 600, 1200, 2400, 4800, 6000]
Ts = [0.1, 0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125]
N = [60, 120, 240, 480, 600, 1200, 2400, 4800]
seq_time_means = []
seq_time_medians = []
par_time_means = []
par_time_medians = []

for sampling_period, horizon in zip(Ts, N):
    seq_time_Ts = []
    par_time_Ts = []
    downsampling = 1
    dynamics = discretize_dynamics(ode, sampling_period, downsampling)

    x0 = jnp.array([2.0, 1.0])
    key = jax.random.PRNGKey(1)
    u = 0. * jax.random.normal(key, shape=(horizon, 1))
    linear_problem = OCP(dynamics, constraints, stage_cost, final_cost, total_cost)

    anon_par_log_barrier = lambda u, x0: par_log_barrier(linear_problem, u, x0)
    anon_seq_log_barrier = lambda u, x0: seq_log_barrier(linear_problem, u, x0)
    _jitted_par_log_barrier = jax.jit(anon_par_log_barrier)
    _jitted_seq_log_barrier = jax.jit(anon_seq_log_barrier)
    _, _ = _jitted_par_log_barrier(u, x0)
    _, _ = _jitted_seq_log_barrier(u, x0)
    for i in range(10):
        print(i)

        start = time.time()
        par_x, par_u = _jitted_par_log_barrier(u, x0)
        jax.block_until_ready(par_x)
        end = time.time()
        par_log_time = end - start
        print('par finished')

        start = time.time()
        seq_x, seq_u = _jitted_seq_log_barrier(u, x0)
        jax.block_until_ready(seq_x)
        end = time.time()
        seq_log_time = end - start
        print('seq finished')

        seq_time_Ts.append(seq_log_time)
        par_time_Ts.append(par_log_time)

    seq_time_means.append(jnp.mean(jnp.array(seq_time_Ts)))
    seq_time_medians.append(jnp.median(jnp.array(seq_time_Ts)))
    par_time_means.append(jnp.mean(jnp.array(par_time_Ts)))
    par_time_medians.append(jnp.median(jnp.array(par_time_Ts)))

seq_time_means_arr = jnp.array(seq_time_means)
seq_time_medians_arr = jnp.array(seq_time_medians)
par_time_means_arr = jnp.array(par_time_means)
par_time_medians_arr = jnp.array(par_time_medians)

df_mean_seq = pd.DataFrame(seq_time_means_arr)
df_median_seq = pd.DataFrame(seq_time_medians_arr)
df_mean_par = pd.DataFrame(par_time_means_arr)
df_median_par = pd.DataFrame(par_time_medians_arr)

df_mean_seq.to_csv("log_seq_means.csv")
df_median_seq.to_csv("log_seq_median.csv")
df_mean_par.to_csv("log_par_means.csv")
df_median_par.to_csv("log_par_median.csv")


