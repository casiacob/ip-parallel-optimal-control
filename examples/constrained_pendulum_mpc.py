import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
from noc.seq_log_barrier_optimal_control import seq_log_barrier
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


dsic_step = 0.005
downsampling = 1
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=dsic_step, downsampling=downsampling
)

horizon = 140
key = jax.random.PRNGKey(1)
u0 = 0.1 * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.1), -0.1])
ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)

def seq_mpc_loop(carry, input):
    prev_x, prev_u = carry
    x, u = seq_log_barrier(ilqr, prev_u, prev_x)
    return (x[1], u), (x[1], u[0])

def par_mpc_loop(carry, input):
    prev_x, prev_u = carry
    x, u = par_log_barrier(ilqr, prev_u, prev_x)
    return (x[1], u), (x[1], u[0])

_jitted_seq_mpc_loop = jax.jit(seq_mpc_loop)
_jitted_par_mpc_loop = jax.jit(par_mpc_loop)



_, (mpc_x, mpc_u) = jax.lax.scan(_jitted_seq_mpc_loop, (x0, u0), xs=None, length=800)
start = time.time()
_, (mpc_x_seq, mpc_u_seq) = jax.lax.scan(_jitted_seq_mpc_loop, (x0, u0), xs=None, length=800)
jax.block_until_ready(mpc_u_seq)
end = time.time()
seq_time = end - start

_, (mpc_x, mpc_u) = jax.lax.scan(_jitted_par_mpc_loop, (x0, u0), xs=None, length=800)
start = time.time()
_, (mpc_x_par, mpc_u_par) = jax.lax.scan(_jitted_par_mpc_loop, (x0, u0), xs=None, length=800)
jax.block_until_ready(mpc_u_par)
end = time.time()
par_time = end - start


print('Sequential time: ', seq_time)
print('Parallel time  : ', par_time)
print('par vs seq solution')
print('u: ', jnp.max(jnp.abs(mpc_u_par - mpc_u_seq)))
print('x_1: ', jnp.max(jnp.abs(mpc_x_par[:, 0] - mpc_x_seq[:, 0])))
print('x_2: ', jnp.max(jnp.abs(mpc_x_par[:, 1] - mpc_x_seq[:, 1])))
#######################################################################################################################

plt.plot(mpc_x_seq[:, 0], label='angle seq')
plt.plot(mpc_x_par[:, 0], label='angle par')
# # plt.show()
plt.plot(mpc_u_seq, label='control seq')
plt.plot(mpc_u_par, label='control par')
plt.legend()
plt.show()