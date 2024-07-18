import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_interior_point_newton import par_interior_point_optimal_control
from noc.differential_dynamic_programming import interior_point_ddp
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics, euler
from jax import lax, debug
from noc.utils import wrap_angle

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")


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
    # err = state - goal_state

    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action, bp):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    # state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    # action_cost = jnp.diag(jnp.array([1e-3]))
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


# def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
#     #roulet
#     gravity = 10.0
#     length = 1.0
#     mass = 1.0
#     damping = 1e-2
#
#     position, velocity = state
#     return jnp.hstack(
#         (
#             velocity,
#             -gravity / length * jnp.sin(position)
#             + (action - damping * velocity) / (mass * length**2),
#         )
#     )
#


simulation_step = 0.01
downsampling = 1
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=simulation_step, downsampling=downsampling
)
# dynamics = euler(pendulum, simulation_step)

horizon = 300
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.1), -0.1])
nonlinear_problem = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
reg_scheme = jnp.bool_(1.0)
x_nonlin_rollout, u_nonlin_rollour, nonlin_rollout_iterations = (
    par_interior_point_optimal_control(
        nonlinear_problem, u, x0, jnp.bool_(1.0), reg_scheme
    )
)
x_N, u_N, N_iterations = par_interior_point_optimal_control(
    nonlinear_problem, u, x0, jnp.bool_(0.0), reg_scheme
)
x_ddp, u_ddp, ddp_iterations = interior_point_ddp(nonlinear_problem, u, x0)
print("nonlin rollout : ", nonlin_rollout_iterations)
print("N              : ", N_iterations)
print("ddp            : ", ddp_iterations)
plt.plot(x_nonlin_rollout[:, 0])
plt.plot(x_N[:, 0])
plt.plot(x_ddp[:, 0])
plt.show()
plt.plot(u_nonlin_rollour)
plt.plot(u_N)
plt.plot(u_ddp)
plt.show()
