import jax.numpy as jnp
import jax.scipy as jcp
from jax import grad, jacrev, lax, hessian
import jax
from noc.optimal_control_problem import OCP, Derivatives
from typing import Callable
from noc.utils import rollout


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


def bwd_pass(
    final_cost: Callable,
    final_state: jnp.ndarray,
    d: Derivatives,
    reg_param: float,
):
    grad_cost_norm = jnp.linalg.norm(d.cu)
    reg_param = reg_param * grad_cost_norm

    def body(carry, inp):
        Vx, Vxx = carry
        cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu = inp

        Qx = cx + fx.T @ Vx
        Qu = cu + fu.T @ Vx
        Qxx = cxx + fx.T @ Vxx @ fx + jnp.tensordot(Vx, fxx, axes=1)
        Qxu = cxu + fx.T @ Vxx @ fu + jnp.tensordot(Vx, fxu, axes=1)
        Quu = cuu + fu.T @ Vxx @ fu + jnp.tensordot(Vx, fuu, axes=1)
        Quu = Quu + reg_param * jnp.eye(Quu.shape[0])
        eig_vals, _ = jnp.linalg.eigh(Quu)
        pos_def = jnp.all(eig_vals > 0)

        k = -jcp.linalg.solve(Quu, Qu)
        K = -jcp.linalg.solve(Quu, Qxu.T)

        dV = -0.5 * Qu @ jcp.linalg.solve(Quu, Qu)
        Vx = Qx - Qu @ jcp.linalg.solve(Quu, Qxu.T)
        Vxx = Qxx - Qxu @ jcp.linalg.solve(Quu, Qxu.T)
        return (Vx, Vxx), (k, K, dV, pos_def, Qu)

    Vx_final = grad(final_cost)(final_state)
    Vxx_final = hessian(final_cost)(final_state)

    _, (ffgain, gain, cost_diff, feasible_bwd_pass, Hu) = jax.lax.scan(
        body,
        (Vx_final, Vxx_final),
        (d.cx, d.cu, d.cxx, d.cuu, d.cxu, d.fx, d.fu, d.fxx, d.fuu, d.fxu),
        reverse=True,
    )
    pred_reduction = jnp.sum(cost_diff)
    feasible_bwd_pass = jnp.all(feasible_bwd_pass)

    return ffgain, gain, pred_reduction, feasible_bwd_pass, Hu


def nonlin_rollout(
    ocp: OCP,
    gain: jnp.ndarray,
    ffgain: jnp.ndarray,
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
):
    def body(x_hat, inp):
        K, k, x, u = inp
        u_hat = u + k + K @ (x_hat - x)
        next_x_hat = ocp.dynamics(x_hat, u_hat)
        return next_x_hat, (x_hat, u_hat)

    new_final_state, (new_states, new_controls) = jax.lax.scan(
        body, nominal_states[0], (gain, ffgain, nominal_states[:-1], nominal_controls)
    )
    new_states = jnp.vstack((new_states, new_final_state))
    return new_states, new_controls


def check_feasibility(ocp: OCP, x: jnp.ndarray, u: jnp.ndarray):
    cons = jax.vmap(ocp.constraints)(x[:-1], u)
    return jnp.all(cons <= 0)


def ddp(
    ocp: OCP, controls: jnp.ndarray, initial_state: jnp.ndarray, barrier_param: float
):
    states = rollout(ocp.dynamics, controls, initial_state)
    initial_reg_param = 1.0
    initial_reg_inc = 2.0

    def while_body(val):
        x, u, iterations, reg_param, reg_inc, _, _ = val
        # jax.debug.print("Iteration:    {x}", x=iterations)

        cost = ocp.total_cost(x, u, barrier_param)
        # jax.debug.print("cost:         {x}", x=cost)

        d = compute_derivatives(ocp, x, u, barrier_param)

        ffgain, gain, pred_reduction, feasible_bwd_pass, Hu = bwd_pass(
            ocp.final_cost, x[-1], d, reg_param
        )
        temp_x, temp_u = nonlin_rollout(ocp, gain, ffgain, x, u)

        Hu_norm = jnp.linalg.norm(Hu)
        new_traj_feasible = check_feasibility(ocp, temp_x, temp_u)
        new_cost = jnp.where(
            new_traj_feasible, ocp.total_cost(temp_x, temp_u, barrier_param), jnp.inf
        )
        # jax.debug.print("new cost:     {x}", x=new_cost)
        # jax.debug.print("bp feasible:  {x}", x=feasible_bwd_pass)

        actual_reduction = new_cost - cost
        gain_ratio = actual_reduction / pred_reduction
        # jax.debug.print("gain ratio:   {x}", x=gain_ratio)
        accept_cond = jnp.logical_and(gain_ratio > 0, feasible_bwd_pass)
        reg_param = jnp.where(
            accept_cond,
            reg_param * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
            reg_param * reg_inc,
        )
        reg_param = jnp.clip(reg_param, 1e-16, 1e16)
        reg_inc = jnp.where(accept_cond, 2.0, 2 * reg_inc)
        x = jnp.where(accept_cond, temp_x, x)
        u = jnp.where(accept_cond, temp_u, u)
        # jax.debug.print("accept:       {x}", x=accept_cond)
        # jax.debug.print("|H_u|:        {x}", x=Hu_norm)

        iterations = iterations + 1
        # jax.debug.print("---------------------------------")
        # jax.debug.breakpoint()
        return x, u, iterations, reg_param, reg_inc, Hu_norm, feasible_bwd_pass

    def while_cond(val):
        _, _, iterations, _, _, Hu_norm, bp_feasible = val
        exit_cond = jnp.logical_and(Hu_norm < 1e-4, bp_feasible)
        exit_cond = jnp.logical_or(exit_cond, iterations > 1000)
        # jax.debug.breakpoint()
        return jnp.logical_not(exit_cond)

    (opt_states, opt_controls, total_iterations, _, _, _, _) = lax.while_loop(
        while_cond,
        while_body,
        (
            states,
            controls,
            0,
            initial_reg_param,
            initial_reg_inc,
            jnp.array(1.0),
            jnp.bool_(1.0),
        ),
    )

    return opt_states, opt_controls, total_iterations


def interior_point_ddp(ocp: OCP, controls: jnp.ndarray, initial_state: jnp.ndarray):
    barrier_param = 0.1

    def while_body(val):
        u, bp, t = val
        _, u, newton_iterations = ddp(ocp, u, initial_state, bp)
        bp = bp / 5
        t = t + newton_iterations
        # jax.debug.breakpoint()
        return u, bp, t

    def while_cond(val):
        _, bp, _ = val
        return bp > 1e-4

    opt_u, _, N_iterations = lax.while_loop(
        while_cond, while_body, (controls, barrier_param, 0)
    )
    # jax.debug.print("converged in {x}", x=t_conv)
    opt_x = rollout(ocp.dynamics, opt_u, initial_state)
    optimal_cost = ocp.total_cost(opt_x, opt_u, 0.0)
    jax.debug.print("optimal cost {x}", x=optimal_cost)
    return opt_x, opt_u, N_iterations
