import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.newton_oc import cnoc
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
from jax import lax, debug
from noc.utils import wrap_angle
import time

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")
def constraints(state, control):
    control_ub = 5.
    control_lb = -5.

    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state):
    goal_state = jnp.array((jnp.pi, 0.0))
    final_state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    # err = state - goal_state

    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action, bp):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    # err = state - goal_state
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


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=simulation_step, downsampling=downsampling
)

horizon = 15
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u0 = sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.1), -0.1])
ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)

def mpc_loop(carry, input):
    prev_x, prev_u = carry
    x, u = cnoc(ilqr, prev_u, prev_x)
    return (x[1], u), (x[1], u[0])

jitted_mpc_loop = jax.jit(mpc_loop)
_, (mpc_x, mpc_u) = jax.lax.scan(jitted_mpc_loop, (x0, u0), xs=None, length=80)
start = time.time()
_, (mpc_x, mpc_u) = jax.lax.scan(jitted_mpc_loop, (x0, u0), xs=None, length=80)
jax.block_until_ready(mpc_u)
end = time.time()
print(end-start)

# mpc_x = jnp.vstack((x0, mpc_x))
# # x_noc, u_noc = cnoc(ilqr, u, x0)
plt.plot(mpc_x[:, 0])
plt.plot(mpc_x[:, 1])
# # plt.show()
plt.plot(mpc_u)
plt.show()