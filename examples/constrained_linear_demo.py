import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
from noc.seq_log_barrier_optimal_control import seq_log_barrier
from noc.utils import discretize_dynamics, rollout
import jax
import time
import argparse

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

config.update("jax_platform_name", "gpu")

parser = argparse.ArgumentParser(description='Configure par vs seq experiment')
parser.add_argument('--parallel', action=argparse.BooleanOptionalAction, help='use parallel or sequential solver')
parser.add_argument('--Ts', metavar='Ts', type=float, default=0.1, help='sampling period')
parser.add_argument('--N', metavar='N', type=int, default=60, help='horizon')
args = parser.parse_args()

def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


sampling_period = args.Ts
downsampling = 1
dynamics = discretize_dynamics(ode, sampling_period, downsampling)

horizon = args.N

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


x0 = jnp.array([2.0, 1.0])
key = jax.random.PRNGKey(1)
u = 0. * jax.random.normal(key, shape=(horizon, 1))
linear_problem = OCP(dynamics, constraints, stage_cost, final_cost, total_cost)

anon_par_log_barrier = lambda u, x0: par_log_barrier(linear_problem, u, x0)
anon_seq_log_barrier = lambda u, x0: seq_log_barrier(linear_problem, u, x0)
_jitted_par_log_barrier = jax.jit(anon_par_log_barrier)
_jitted_seq_log_barrier = jax.jit(anon_seq_log_barrier)

_, _ = _jitted_par_log_barrier( u, x0)
start = time.time()
par_x, par_u = _jitted_par_log_barrier( u, x0)
jax.block_until_ready(par_x)
end = time.time()
par_time = end - start


_, _ = _jitted_seq_log_barrier( u, x0)
start = time.time()
seq_x, seq_u = _jitted_seq_log_barrier( u, x0)
jax.block_until_ready(seq_x)
end = time.time()
seq_time = end - start

print(sampling_period, ", ", par_time, ", ", seq_time)