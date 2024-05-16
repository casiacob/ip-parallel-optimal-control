import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.newton_oc import cnoc
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics, rollout
from jax import lax, debug
from noc.utils import wrap_angle
from noc.newton_oc import compute_derivatives, compute_Lagrange_multipliers, compute_lqr_params, noc_to_lqt
from paroc import par_bwd_pass, par_fwd_pass
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

horizon = 20
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.1), -0.1])
optimal_control_problem = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
x = rollout(optimal_control_problem.dynamics, u, x0)
bp = 0.1
rp = 1.0

anon_compute_derivatives = lambda x, u, bp: compute_derivatives(optimal_control_problem, x, u, bp)
anon_compute_Lagrange_multipliers = lambda x, u, d: compute_Lagrange_multipliers(optimal_control_problem, x, u, d)
_jitted_par_bwd_pass = jax.jit(par_bwd_pass)
_jitted_par_fwd_pass = jax.jit(par_fwd_pass)
_jitted_compute_derivatives = jax.jit(anon_compute_derivatives)
_jitted_compute_Lagrange_multipliers = jax.jit(anon_compute_Lagrange_multipliers)
_jitted_compute_lqr_params = jax.jit(compute_lqr_params)
_jitted_noc_to_lqt = jax.jit(noc_to_lqt)

d = _jitted_compute_derivatives(x, u, bp)
l = _jitted_compute_Lagrange_multipliers(x, u, d)
ru, Q, R, M = _jitted_compute_lqr_params(l, d)
R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), rp * jnp.eye(R.shape[1]))
lqt = _jitted_noc_to_lqt(ru, Q, R, M, d.fx, d.fu)
Kx_par, d_par, S_par, v_par, pred_reduction, convex_problem = _jitted_par_bwd_pass(lqt)
du_par, dx_par = _jitted_par_fwd_pass(lqt, jnp.zeros(x[0].shape[0]), Kx_par, d_par)


start = time.time()
d = _jitted_compute_derivatives(x, u, bp)
jax.block_until_ready(d)
end = time.time()
print('derivatives', end-start)

start = time.time()
l = _jitted_compute_Lagrange_multipliers(x, u, d)
jax.block_until_ready(l)
end = time.time()
print('lagrange', end-start)

start = time.time()
ru, Q, R, M = _jitted_compute_lqr_params(l, d)
jax.block_until_ready(ru)
end = time.time()
print('lqr_params', end-start)


start = time.time()
R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), rp * jnp.eye(R.shape[1]))
jax.block_until_ready(R)
end = time.time()
print('R reg kron', end-start)

start = time.time()
lqt = _jitted_noc_to_lqt(ru, Q, R, M, d.fx, d.fu)
jax.block_until_ready(lqt)
end = time.time()
print('lqt', end-start)

Kx_par, d_par, S_par, v_par, pred_reduction, convex_problem = _jitted_par_bwd_pass(lqt)
jax.block_until_ready(Kx_par)
end = time.time()
print('bwd', end-start)

start = time.time()
du_par, dx_par = _jitted_par_fwd_pass(lqt, jnp.zeros(x[0].shape[0]), Kx_par, d_par)
jax.block_until_ready(du_par)
end = time.time()
print('fwd', end-start)