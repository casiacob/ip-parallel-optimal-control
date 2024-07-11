import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.par_log_barrier_optimal_control import par_log_barrier
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
from jax import lax, debug
from noc.utils import wrap_angle

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")

def constraints(state, control):
    control_ub = 10.
    control_lb = -10.

    c0 = control[0] - control_ub
    c1 = -control[0] + control_lb
    c2 = control[1] - control_ub
    c3 = -control[1] + control_lb
    return jnp.hstack((c0, c1, c2, c3))


def final_cost(state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([2e1, 2e1, 1e-1, 1e-1]))
    goal_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c

def transient_cost(state: jnp.ndarray, action: jnp.ndarray, bp) -> float:
    goal_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    state_cost = jnp.diag(jnp.array([1e1, 1e-1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3, 1e-3]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier

def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)

def double_pendulum(
    state: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:

    # https://underactuated.mit.edu/multibody.html#section1

    g = 9.81
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    k1, k2 = 1e-3, 1e-3

    th1, th2, dth1, dth2 = state
    u1, u2 = action

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    s12 = jnp.sin(th1 + th2)

    # inertia
    M = jnp.array(
        [
            [
                (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * c2,
                m2 * l2**2 + m2 * l1 * l2 * c2,
            ],
            [
                m2 * l2**2 + m2 * l1 * l2 * c2,
                m2 * l2**2
            ],
        ]
    )

    # Corliolis
    C = jnp.array(
        [
            [
                0.0,
                -m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2
            ],
            [
                0.5 * m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2,
                -0.5 * m2 * l1 * l2 * dth1 * s2,
            ],
        ]
    )

    # gravity
    tau = -g * jnp.array(
        [
            (m1 + m2) * l1 * s1 + m2 * l2 * s12,
            m2 * l2 * s12
        ]
    )

    B = jnp.eye(2)

    u1 = u1 - k1 * dth1
    u2 = u2 - k2 * dth2

    u = jnp.hstack([u1, u2])
    v = jnp.hstack([dth1, dth2])

    a = jnp.linalg.solve(M, tau + B @ u - C @ v)

    return jnp.hstack((v, a))


simulation_step = 0.005
downsampling = 1
dynamics = discretize_dynamics(
    ode=double_pendulum, simulation_step=simulation_step, downsampling=downsampling
)

horizon = 140
sigma = jnp.array([0.01])
key = jax.random.PRNGKey(271)
u0 = sigma * jax.random.normal(key, shape=(horizon, 2))
x0 = jnp.array(
    [
        wrap_angle(-0.01),
        wrap_angle(0.01),
        -0.01,
        0.01,
    ]
)

ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
# # x_noc, u_noc = par_log_barrier(ilqr, u, x0)
# plt.plot(x_noc[:, 0])
# plt.plot(x_noc[:, 1])
# # plt.show()
# plt.plot(u_noc)
# plt.show()

def par_mpc_loop(carry, input):
    prev_x, prev_u = carry
    x, u = par_log_barrier(ilqr, prev_u, prev_x)
    jax.debug.print('-----------------------------')
    return (x[1], u), (x[1], u[0])

_jitted_par_mpc_loop = jax.jit(par_mpc_loop)
_, (mpc_x_par, mpc_u_par) = jax.lax.scan(_jitted_par_mpc_loop, (x0, u0), xs=None, length=800)
plt.plot(mpc_x_par[:, 0])
plt.plot(mpc_x_par[:, 1])
plt.show()
plt.plot(mpc_u_par[:, 0])
plt.plot(mpc_u_par[:, 1])
plt.show()