import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
from noc.par_log_barrier_optimal_control_gauss_newton import gn_par_log_barrier
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
from jax import lax, debug
from noc.utils import wrap_angle

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")

def constraints(state, control):
    control_ub = 50.
    control_lb = -50.

    c0 = control[0] - control_ub
    c1 = -control[0] + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (
        0.5
        * (_wrapped - goal_state).T
        @ final_state_cost
        @ (_wrapped - goal_state)
    )
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray, bp) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))
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


def cartpole(
    state: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:

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
        action
        + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )

simulation_step = 0.01
downsampling = 1
dynamics = discretize_dynamics(
    ode=cartpole, simulation_step=simulation_step, downsampling=downsampling
)

horizon = 75
key = jax.random.PRNGKey(271)
u0 = jnp.array([0.01]) * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([0.01, wrap_angle(-0.01), 0.01, -0.01])


ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
def par_mpc_loop(carry, input):
    prev_x, prev_u = carry
    x, u, iterations = par_log_barrier(ilqr, prev_u, prev_x)
    jax.debug.print('-----------------------------')
    return (x[1], u), (x[1], u[0], iterations)

_jitted_par_mpc_loop = jax.jit(par_mpc_loop)
_, (mpc_x_par, mpc_u_par, iterations) = jax.lax.scan(_jitted_par_mpc_loop, (x0, u0), xs=None, length=500)
plt.plot(mpc_x_par[:, 0])
plt.plot(mpc_x_par[:, 1])
plt.show()
plt.plot(mpc_u_par[:, 0])
plt.show()
print(jnp.sum(iterations))