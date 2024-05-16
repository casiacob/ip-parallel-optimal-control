import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
from noc.utils import discretize_dynamics
import jax
from paroc.lqt_problem import LQT
from paroc import par_bwd_pass, par_fwd_pass
from paroc import seq_bwd_pass, seq_fwd_pass
import time

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cuda")

_jitted_par_bwd_pass = jax.jit(par_bwd_pass, backend='gpu')
_jitted_par_fwd_pass = jax.jit(par_fwd_pass, backend='gpu')
_jitted_seq_bwd_pass = jax.jit(seq_bwd_pass, backend='gpu')
_jitted_seq_fwd_pass = jax.jit(seq_fwd_pass, backend='gpu')

def ode(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ state + B @ control


step = 0.001
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)

x0 = jnp.array([2.0, 1.0])


A = jax.jacfwd(dynamics, 0)(x0, jnp.array([0.]))
B = jax.jacfwd(dynamics, 1)(x0, jnp.array([0.]))


T = 5
mpc_sim_steps = 5000

nx = 2
nu = 1
Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(nu)
P = jnp.diag(jnp.array([1e2, 1e0]))

r = jnp.zeros((T, nx))
s = jnp.zeros((T, nu))
HT = jnp.eye(nx)
H = jnp.kron(jnp.ones((T, 1, 1)), HT)
Q = jnp.kron(jnp.ones((T, 1, 1)), Q)
R = jnp.kron(jnp.ones((T, 1, 1)), R)
A = jnp.kron(jnp.ones((T, 1, 1)), A)
B = jnp.kron(jnp.ones((T, 1, 1)), B)
Z = jnp.kron(jnp.ones((T, 1, 1)), jnp.eye(nu))
XT = P
rT = jnp.zeros(nx)
c = jnp.zeros((T, nx))
M = jnp.zeros((nx, nu))
M = jnp.kron(jnp.ones((T, 1, 1)), M)
lqt = LQT(A, B, c, XT, HT, rT, Q, H, r, R, Z, s, M)



def par_mpc_loop(prev_x, inp):
    Kx_par, d_par, _, _, _, _ = _jitted_par_bwd_pass(lqt)
    u_par, x_par = _jitted_par_fwd_pass(lqt, prev_x, Kx_par, d_par)
    return x_par[1], (x_par[1], u_par[0])

def seq_mpc_loop(prev_x, inp):
    Kx_seq, d_seq, _, _ = _jitted_seq_bwd_pass(lqt)
    u_seq, x_seq = _jitted_seq_fwd_pass(lqt, prev_x, Kx_seq, d_seq)
    return x_seq[1], (x_seq[1], u_seq[0])

_, (mpc_x_par, mpc_u_par) = jax.lax.scan(par_mpc_loop, x0, xs=None, length=mpc_sim_steps)
_, (mpc_x_seq, mpc_u_seq) = jax.lax.scan(seq_mpc_loop, x0, xs=None, length=mpc_sim_steps)

start = time.time()
_, (mpc_x_par, mpc_u_par) = jax.lax.scan(par_mpc_loop, x0, xs=None, length=mpc_sim_steps)
jax.block_until_ready(mpc_x_par)
end = time.time()
print(end-start)

start = time.time()
_, (mpc_x_seq, mpc_u_seq) = jax.lax.scan(seq_mpc_loop, x0, xs=None, length=mpc_sim_steps)
jax.block_until_ready(mpc_x_seq)
end = time.time()
print(end-start)

plt.plot(mpc_x_seq[:, 0])
plt.plot(mpc_x_par[:, 0])
plt.show()