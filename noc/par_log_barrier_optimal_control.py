import jax.numpy as jnp
from jax import grad, jacrev, lax, hessian
import jax
from noc.optimal_control_problem import OCP, Derivatives, LinearizedOCP
from paroc import par_bwd_pass, par_fwd_pass
from paroc.lqt_problem import LQT
from noc.utils import rollout
from typing import Callable
from noc.costates import par_costates


def compute_derivatives(
    ocp: OCP, states: jnp.ndarray, controls: jnp.ndarray, bp: float
):
    def body(x, u):
        cx_k, cu_k = grad(ocp.stage_cost, (0, 1))(x, u, bp)
        cxx_k = hessian(ocp.stage_cost, 0)(x, u, bp)
        cuu_k = hessian(ocp.stage_cost, 1)(x, u, bp)
        cxu_k = jacrev(jacrev(ocp.stage_cost, 0), 1)(x, u, bp)
        fx_k, fu_k = jacrev(ocp.dynamics, (0, 1))(x, u)
        fxx_k = jacrev(jacrev(ocp.dynamics, 0), 0)(x, u)
        fuu_k = jacrev(jacrev(ocp.dynamics, 1), 1)(x, u)
        fxu_k = jacrev(jacrev(ocp.dynamics, 0), 1)(x, u)
        return cx_k, cu_k, cxx_k, cuu_k, cxu_k, fx_k, fu_k, fxx_k, fuu_k, fxu_k

    cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu = jax.vmap(body)(states[:-1], controls)
    return Derivatives(cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu)


def compute_lqr_params(lagrange_multipliers: jnp.ndarray, d: Derivatives):
    def body(l, cu, cxx, cuu, cxu, fu, fxx, fuu, fxu):
        # lqr params
        ru = cu + fu.T @ l
        Q = cxx + jnp.tensordot(l, fxx, axes=1)
        R = cuu + jnp.tensordot(l, fuu, axes=1)
        M = cxu + jnp.tensordot(l, fxu, axes=1)
        return ru, Q, R, M

    return jax.vmap(body)(
        lagrange_multipliers[1:], d.cu, d.cxx, d.cuu, d.cxu, d.fu, d.fxx, d.fuu, d.fxu
    )


def check_feasibility(ocp: OCP, x: jnp.ndarray, u: jnp.ndarray):
    cons = jax.vmap(ocp.constraints)(x[:-1], u)
    return jnp.all(cons <= 0)


def noc_to_lqt(
    ru: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    M: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
):
    T = Q.shape[0]
    nx = Q.shape[1]
    nu = R.shape[1]

    def references(X_t, U_t, M_t, ru_t):
        X_inv_M = jnp.linalg.solve(X_t, M_t)
        s_t = -jnp.linalg.solve(U_t - M_t.T @ X_inv_M, ru_t)
        r_t = -X_inv_M @ s_t
        return r_t, s_t

    r, s = jax.vmap(references)(Q, R, M, ru)
    H = jnp.eye(nx)
    HT = H
    H = jnp.kron(jnp.ones((T, 1, 1)), H)
    Z = jnp.eye(nu)
    Z = jnp.kron(jnp.ones((T, 1, 1)), Z)
    XT = Q[0]
    rT = jnp.zeros(nx)
    c = jnp.zeros((T, nx))
    lqt = LQT(A, B, c, XT, HT, rT, Q, H, r, R, Z, s, M)
    return lqt


def par_solution(ocp: OCP, x: jnp.ndarray, u: jnp.ndarray, bp: float, rp: float):
    d = compute_derivatives(ocp, x, u, bp)
    l = par_costates(ocp, x[-1], d)
    ru, Q, R, M = compute_lqr_params(l, d)
    R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), rp * jnp.eye(R.shape[1]))
    lqt = noc_to_lqt(ru, Q, R, M, d.fx, d.fu)
    Kx_par, d_par, S_par, v_par, pred_reduction, convex_problem = par_bwd_pass(lqt)
    du_par, dx_par = par_fwd_pass(lqt, jnp.zeros(x[0].shape[0]), Kx_par, d_par)
    # jax.debug.breakpoint()
    return dx_par, du_par, pred_reduction, convex_problem, ru


def noc(ocp: OCP, controls: jnp.ndarray, initial_state: jnp.ndarray, bp: float):
    states = rollout(ocp.dynamics, controls, initial_state)
    mu0 = 1.0
    nu0 = 2.0

    def while_body(val):
        x, u, t, mu, nu, _, _ = val
        # jax.debug.print("Iteration:    {x}", x=t)

        cost = ocp.total_cost(x, u, bp)
        # jax.debug.print("cost:         {x}", x=cost)
        # jax.debug.breakpoint()

        dx, du, predicted_reduction, bp_feasible, Hu = par_solution(ocp, x, u, bp, mu)
        Hu_norm = jnp.max(jnp.abs(Hu))

        temp_u = u + du
        temp_x = x + dx
        new_traj_feasible = check_feasibility(ocp, temp_x, temp_u)
        new_cost = jnp.where(
            new_traj_feasible, ocp.total_cost(temp_x, temp_u, bp), jnp.inf
        )
        # jax.debug.print("new cost:     {x}", x=new_cost)
        # jax.debug.print("bp feasible:  {x}", x=bp_feasible)

        actual_reduction = new_cost - cost
        gain_ratio = actual_reduction / predicted_reduction
        # jax.debug.print("gain ratio:   {x}", x=gain_ratio)

        accept_cond = jnp.logical_and(gain_ratio > 0, bp_feasible)
        # jax.debug.print("Accept cond:  {x}", x=accept_cond)
        mu = jnp.where(
            accept_cond,
            mu * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
            mu * nu,
        )
        nu = jnp.where(accept_cond, 2.0, 2 * nu)
        x = jnp.where(accept_cond, temp_x, x)
        u = jnp.where(accept_cond, temp_u, u)
        # jax.debug.print("reg param:    {x}", x=mu)
        # jax.debug.print("a red:        {x}", x=actual_reduction)
        # jax.debug.print("p red         {x}", x=predicted_reduction)
        # jax.debug.print("|H_u|:        {x}", x=Hu_norm)

        t = t + 1
        # jax.debug.print("---------------------------------")
        # jax.debug.breakpoint()
        return x, u, t, mu, nu, Hu_norm, bp_feasible

    def while_cond(val):
        _, _, t, _, _, Hu_norm, bp_feasible = val
        exit_cond = jnp.logical_and(Hu_norm < 1e-4, bp_feasible)
        # exit_cond = jnp.logical_or(exit_cond, t > 1)
        return jnp.logical_not(exit_cond)

    (
        opt_x,
        opt_u,
        _,
        _,
        _,
        _,
        _,
    ) = lax.while_loop(
        while_cond,
        while_body,
        (states, controls, 0, mu0, nu0, jnp.array(1.0), jnp.bool(1.0)),
    )
    # jax.debug.breakpoint()
    return opt_x, opt_u


def par_log_barrier(ocp: OCP, controls: jnp.ndarray, initial_state: jnp.ndarray):
    barrier_param = 0.1

    def while_body(val):
        u, bp, t = val
        _, u = noc(ocp, u, initial_state, bp)
        bp = bp / 5
        t = t + 1
        # jax.debug.breakpoint()
        return u, bp, t

    def while_cond(val):
        _, bp, t = val
        return bp < 1e-4

    opt_u, _, t_conv = lax.while_loop(
        while_cond, while_body, (controls, barrier_param, 0)
    )
    # jax.debug.print("converged in {x}", x=t_conv)
    opt_x = rollout(ocp.dynamics, opt_u, initial_state)
    optimal_cost = ocp.total_cost(opt_x, opt_u, 0.0)
    # jax.debug.print("optimal cost {x}", x=optimal_cost)
    return opt_x, opt_u
