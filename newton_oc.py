import jax.numpy as jnp
from jax import grad, jacfwd, lax, hessian
import jax
from typing import Callable


def compute_states(
    dynamics: Callable, controls: jnp.ndarray, initial_state: jnp.ndarray
):
    def body(xt, ut):
        return dynamics(xt, ut), dynamics(xt, ut)

    _, states = lax.scan(body, initial_state, controls)
    states = jnp.vstack((initial_state, states))
    return states


def compute_Lagrange_multipliers(ocp, states, controls):
    xT = states[-1]
    lamda_T = grad(ocp.final_cost, 0)(xT)

    def body(carry, inp):
        lamda = carry
        x, u = inp
        cx = grad(ocp.stage_cost, 0)(x, u)
        fx = jacfwd(ocp.dynamics, 0)(x, u)
        lamda = cx + fx @ lamda
        return lamda, lamda

    _, lamda_opt = lax.scan(body, lamda_T, (states[:-1], controls), reverse=True)
    lamda_opt = jnp.vstack((lamda_opt, lamda_T))
    return lamda_opt


def bwd_pass(ocp, states, controls, lamdas):
    def bwd_step(carry, inp):
        Vxx, Vx = carry
        x, u, lamda = inp
        fx = jacfwd(ocp.dynamics, 0)(x, u)
        fu = jacfwd(ocp.dynamics, 1)(x, u)
        fxx = jacfwd(jacfwd(ocp.dynamics, 0), 0)(x, u)
        fuu = jacfwd(jacfwd(ocp.dynamics, 1), 1)(x, u)
        fxu = jacfwd(jacfwd(ocp.dynamics, 0), 1)(x, u)
        cu = grad(ocp.stage_cost, 1)(x, u)
        cxx = hessian(ocp.stage_cost, 0)(x, u)
        cuu = hessian(ocp.stage_cost, 1)(x, u)
        cxu = jacfwd(jacfwd(ocp.stage_cost, 0), 1)(x, u)

        ru = cu + lamda.T @ fu
        Q = cxx + jnp.tensordot(lamda.T, fxx, axes=1)
        R = cuu + jnp.tensordot(lamda.T, fuu, axes=1)
        M = cxu + jnp.tensordot(lamda.T, fxu, axes=1)

        Qxx = Q + fx.T @ Vxx @ fx
        Quu = R + fu.T @ Vxx @ fu
        # Quu = Quu + reg_param * jnp.eye(Quu.shape[0])
        Qxu = M + fx.T @ Vxx @ fu
        Qu = ru + fu.T @ Vx
        Qx = fx.T @ Vx

        k = -jnp.linalg.inv(Quu) @ Qu
        K = -jnp.linalg.inv(Quu) @ Qxu.T

        Vx = k.T @ Quu @ K + k.T @ Qxu.T + Qu @ K + Qx
        Vxx = K.T @ Quu @ K + 2 * K.T @ Qxu.T + Qxx
        dV = k.T @ Qu + 0.5 * k.T @ Quu @ k

        return (Vxx, Vx), (K, k, dV)

    VxxN = hessian(ocp.final_cost)(states[-1])
    # VxN = grad(ocp.final_cost)(states[-1])
    _, bwd_pass_out = lax.scan(
        bwd_step,
        (VxxN, jnp.zeros(states.shape[1])),
        (states[:-1], controls, lamdas[1:]),
        reverse=True,
    )
    Kx, kx, diff_cost = bwd_pass_out
    return Kx, kx, jnp.sum(diff_cost)


def fwd_pass(ocp, states, controls, gain, ffgain):
    def fwd_step(carry, inp):
        prev_x = carry
        K, k, x, u = inp
        dx = prev_x - x
        du = K @ dx + k
        u = u + du
        x = ocp.dynamics(prev_x, u)
        return x, (x, u)

    _, fwd_pass_out = lax.scan(
        fwd_step, states[0], (gain, ffgain, states[:-1], controls)
    )
    opt_states, opt_controls = fwd_pass_out
    opt_states = jnp.vstack((states[0], opt_states))
    return opt_states, opt_controls


def noc(ocp, controls, initial_state):
    states = compute_states(ocp.dynamics, controls, initial_state)

    def while_body(val):
        x, u, t = val
        jax.debug.print("Iteration:    {x}", x=t)
        jax.debug.print("cost:         {x}", x=ocp.total_cost(x, u))
        l = compute_Lagrange_multipliers(ocp, x, u)

        K, k, dv = bwd_pass(ocp, x, u, l)

        x, u = fwd_pass(ocp, x, u, K, k)
        jax.debug.print("new cost:     {x}", x=ocp.total_cost(x, u))
        t = t + 1
        # jax.debug.breakpoint()

        return x, u, t

    def while_cond(val):
        _, _, t = val
        return t < 400

    (
        opt_x,
        opt_u,
        _,
    ) = lax.while_loop(while_cond, while_body, (states, controls, 0))

    return opt_x, opt_u
