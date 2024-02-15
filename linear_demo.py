import jax.numpy as jnp
import jax.random
from optimal_control_problem import ocp
from newton_oc import noc
import matplotlib.pyplot as plt
from utils import discretize_dynamics
import jax

jax.config.update("jax_enable_x64", True)


def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


step = 0.1
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)


def stage_cost(state: jnp.ndarray, control: jnp.ndarray):
    X = jnp.diag(jnp.array([1e2, 1e0]))
    U = 1e-1 * jnp.eye(control.shape[0])
    return 0.5 * state.T @ X @ state + 0.5 * control.T @ U @ control


def final_cost(state: jnp.ndarray):
    P = jnp.diag(jnp.array([1e2, 1e0]))
    return 0.5 * state.T @ P @ state


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(stage_cost)(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


horizon = 40
x0 = jnp.array([2.0, 1.0])
key = jax.random.PRNGKey(1)
u = 0.0 * jax.random.normal(key, shape=(horizon, 1))
lqr = ocp(dynamics, stage_cost, final_cost, total_cost)
x_noc, u_noc = noc(lqr, u, x0)

plt.plot(x_noc[:, 0])
plt.plot(x_noc[:, 1])
plt.show()
