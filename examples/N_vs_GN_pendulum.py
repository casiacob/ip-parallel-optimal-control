import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_primal_barr_optimal_control import par_log_barrier
from noc.par_log_barrier_optimal_control_gauss_newton import gn_par_log_barrier
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
from jax import lax, debug
from noc.utils import wrap_angle

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")
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

horizon = 60
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.1), -0.1])
# x0 = jnp.array([0.1, -0.1])
ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
anon_par_log_barrier = lambda u, x0: par_log_barrier(ilqr, u, x0)
anon_gn_par_log_barrier = lambda u, x0: gn_par_log_barrier(ilqr, u, x0)
_jitted_par_log_barrier = jax.jit(anon_par_log_barrier)
_jitted_seq_log_barrier = jax.jit(anon_gn_par_log_barrier)
_, _ = _jitted_par_log_barrier(u, x0)
_, _ = _jitted_gn_par_log_barrier(u, x0)

start = time.time()
par_x, par_u = _jitted_par_log_barrier(u, x0)
jax.block_until_ready(par_x)
end = time.time()
par_log_time = end - start

start = time.time()
gn_par_x, gn_par_u = _jitted_gn_par_log_barrier(u, x0)
jax.block_until_ready(gn_par_x)
end = time.time()
gn_par_log_time = end - start
print('seq finished')