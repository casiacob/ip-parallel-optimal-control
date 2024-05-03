import jax.numpy as jnp
import jax.random
from jax import config
from noc.optimal_control_problem import OCP
from noc.newton_oc import cnoc
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
from jax import lax, debug

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")



def transient_cost(state: jnp.ndarray, control: jnp.ndarray, bp) -> float:
    state_penalty = jnp.diag(jnp.array([1e-16, 1.0, 5.0, 1.0]))
    control_penalty = jnp.diag(jnp.array([1.0, 10.0]))
    ref = jnp.array([0.0, 0.0, 8.0, 0.0])
    c = (state - ref).T @ state_penalty @ (state - ref)
    c = c + control.T @ control_penalty @ control
    log_barrier = jnp.sum(jnp.log(-constraints(state, control)))
    return c * 0.5 - bp*log_barrier


def final_cost(state: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([1e-16, 1.0, 5.0, 1.0]))
    ref = jnp.array([0.0, 0.0, 8.0, 0.0])
    c = (state - ref).T @ state_penalty @ (state - ref)
    return c * 0.5

def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def constraints(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    x_pos, y_pos, _, _ = state
    acc, steering = control

    # ellipse obstacle parameters
    ea = 5.0
    eb = 2.5
    xc = 17.0
    yc = -1.0

    # ellipse constraint
    S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    dxy = jnp.array([x_pos - xc, y_pos - yc])
    c0 = 1 - dxy.T @ S @ dxy

    # S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    # dxy = jnp.array([x_pos - 35, y_pos - 1])
    # c5 = 1 - dxy.T @ S @ dxy

    # control bounds
    a_ub = 1.5
    a_lb = -3
    steering_ub = 0.15
    steering_lb = -0.15

    c1 = acc - a_ub
    c2 = a_lb - acc
    c3 = steering - steering_ub
    c4 = steering_lb - steering
    # c0 = -1
    return jnp.hstack((c0, c1, c2, c3, c4))
    # return -1.


def car(state: jnp.ndarray, control: jnp.ndarray):
    lf = 1.06
    lr = 1.85
    x, y, v, phi = state
    acceleration, steering = control
    beta = jnp.arctan(jnp.tan(steering * (lr/(lf+lr))))
    return jnp.hstack((
        v * jnp.cos(phi + beta),
        v * jnp.sin(phi + beta),
        acceleration,
        v/lr * jnp.sin(beta)
    ))


simulation_step = 0.1
downsampling = 1
dynamics = discretize_dynamics(
    ode=car, simulation_step=simulation_step, downsampling=downsampling
)

N = 70
mean = jnp.array([0.0, 0.0])
sigma = jnp.array([0.0, 0.0])
key = jax.random.PRNGKey(2)
x0 = jnp.array([0.0, 0.0, 1.0, 0.0])
u = mean + sigma * jax.random.normal(key, shape=(N, 2))

ilqr = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)
x_noc, u_noc = cnoc(ilqr, u, x0)
plt.plot(x_noc[:, 0],x_noc[:, 1])
plt.ylim([-10, 10])
cx1 = 17.0
cy1 = -1.0
a = 5.0  # radius on the x-axis
b = 2.5  # radius on the y-axis
t = jnp.linspace(0, 2 * jnp.pi, 150)
plt.plot(cx1 + a * jnp.cos(t), cy1 + b * jnp.sin(t), color="red")
plt.show()
plt.plot(x_noc[:, 2])
plt.show()
plt.plot(x_noc[:, 3])
plt.show()
plt.plot(u_noc[:, 0])
plt.show()
plt.plot(u_noc[:, 1])
plt.show()