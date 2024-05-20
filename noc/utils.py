import jax.numpy as jnp
from typing import Callable
from jax import lax
import numpy as np
from scipy import linalg


def wrap_angle(x: float) -> float:
    # wrap angle between [0, 2*pi]
    return x % (2.0 * jnp.pi)


def runge_kutta(
    state: jnp.ndarray,
    action: jnp.ndarray,
    ode: Callable,
    step: float,
):
    k1 = ode(state, action)
    k2 = ode(state + 0.5 * step * k1, action)
    k3 = ode(state + 0.5 * step * k2, action)
    k4 = ode(state + step * k3, action)
    return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def discretize_dynamics(ode: Callable, simulation_step: float, downsampling: int):
    def dynamics(
        state: jnp.ndarray,
        action: jnp.ndarray,
    ):
        def _step(t, state_t):
            next_state = runge_kutta(
                state_t,
                action,
                ode,
                simulation_step,
            )
            return next_state

        return lax.fori_loop(
            lower=0,
            upper=downsampling,
            body_fun=_step,
            init_val=state,
        )

    return dynamics


def rollout(dynamics, controls, initial_state):
    def body(xt, ut):
        return dynamics(xt, ut), dynamics(xt, ut)

    _, states = lax.scan(body, initial_state, controls)
    states = jnp.vstack((initial_state, states))
    return states



def get_QP_problem(A, B, P, Q, R, N, C_u, uub, ulb):
    m = A.shape[0]
    H = Q
    G = R
    for i in range(N - 2):
        H = linalg.block_diag(H, Q)
    for i in range(N - 1):
        G = linalg.block_diag(G, R)
    H = linalg.block_diag(H, P)

    Abar = np.eye(m)
    for i in range(1, N):
        Abar = np.append(Abar, np.zeros((i * m, m)), axis=1)
        if i == 1:
            next_row = np.append(-A, np.eye(m), axis=1)
        else:
            next_row = np.append(np.zeros((m, (i - 1) * m)), -A, axis=1)
            next_row = np.append(next_row, np.eye(m), axis=1)
        Abar = np.append(Abar, next_row, axis=0)

    invAbar = linalg.solve(Abar, np.eye(Abar.shape[0]))

    Bbar = B
    for i in range(1, N):
        Bbar = linalg.block_diag(Bbar, B)

    c = np.zeros((m * N, m))
    c[0:m][0:m] = A

    boldQ = Bbar.T @ invAbar.T @ H @ invAbar @ Bbar + G
    boldF = Bbar.T @ invAbar.T @ H @ invAbar @ c


    C_U = np.append(C_u, -C_u, axis=0)
    for i in range(N - 1):
        C_U = linalg.block_diag(C_U, np.append(C_u, -C_u, axis=0))

    ulim = np.append(uub, -ulb)
    for i in range(N - 1):
        ulim = np.append(ulim, np.append(uub, -ulb), axis=0)


    boldG = C_U
    boldW = ulim

    return jnp.array(boldQ), jnp.array(boldF), jnp.array(boldG), jnp.array(boldW)