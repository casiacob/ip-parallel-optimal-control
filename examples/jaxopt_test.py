from jaxopt import BoxOSQP
import jax.numpy as jnp
from noc.utils import discretize_dynamics, get_QP_problem
from jax import jacrev
from jax import config
from jaxopt import BoxOSQP

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

config.update("jax_platform_name", "cuda")


def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


step = 0.01
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)


N = 600
QN = jnp.diag(jnp.array([1e2, 1e0]))
Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(1)
x0 = jnp.array([2.0, 1.0])
Ad = jacrev(dynamics, 0)(jnp.array([0., 0.]), jnp.array([0.]))
Bd = jacrev(dynamics, 1)(jnp.array([0., 0.]), jnp.array([0.]))
umin = -2.5
umax = 2.5

H, g, A, u = get_QP_problem(Ad, Bd, QN, Q, R, N, jnp.eye(1), umax, umin)
l = -jnp.inf * jnp.ones(u.shape[0])
qp = BoxOSQP(jit=True)
sol = qp.run(params_obj=(H, g@x0), params_eq=A, params_ineq=(l, u)).params.primal