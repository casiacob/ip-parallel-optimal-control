import jax.numpy as jnp
from jax import random
from jax import config
from jax import vmap
from noc.optimal_control_problem import IPOCP
from noc.utils import discretize_dynamics, rollout
from noc.primal_dual_ip_noc import ip_pd_oc
import matplotlib.pyplot as plt
# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


step = 0.1
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)


def constraints(state: jnp.ndarray, control: jnp.ndarray):
    g0 = control - 2.5
    g1 = -control - 2.5
    return jnp.hstack((g0, g1))


def stage_cost(state: jnp.ndarray, control: jnp.ndarray):
    X = jnp.diag(jnp.array([1e2, 1e0]))
    U = 1e-1 * jnp.eye(control.shape[0])
    c = 0.5 * state.T @ X @ state + 0.5 * control.T @ U @ control
    return c


def final_cost(state: jnp.ndarray):
    P = jnp.diag(jnp.array([1e2, 1e0]))
    return 0.5 * state.T @ P @ state


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = vmap(stage_cost)(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def barrier_stage_cost(
    state: jnp.ndarray, control: jnp.ndarray, slack: jnp.ndarray, bp: float
):
    c = stage_cost(state, control)
    log_barrier = jnp.sum(jnp.log(slack))
    return c - bp * log_barrier


def barrier_total_cost(
    states: jnp.ndarray, controls: jnp.ndarray, slacks: jnp.ndarray, bp: float
):
    ct = vmap(barrier_stage_cost, in_axes=(0, 0, 0, None))(
        states[:-1], controls, slacks, bp
    )
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


horizon = 20
x0 = jnp.array([2.0, 1.0])
key = random.PRNGKey(1)
u = 0.2 * random.normal(key, shape=(horizon, 1))
lqrip = IPOCP(
    dynamics,
    constraints,
    stage_cost,
    final_cost,
    total_cost,
    barrier_stage_cost,
    barrier_total_cost,
)
z = 0.1 * jnp.ones((horizon, 2))
x = rollout(dynamics, u, x0)
s = -vmap(constraints)(x[:-1], u)
# s=z
#
opt_x, opt_u = ip_pd_oc(lqrip, u, z, s, x0)
plt.plot(opt_x[:, 0])
plt.plot(opt_x[:, 1])
plt.plot(opt_u)
plt.show()