import jax.numpy as jnp
import jax.scipy as jcp
from jax import grad, jacrev, lax, hessian
import jax
from noc.optimal_control_problem import OCP, Derivatives
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


def check_traj_feasibility(ocp: OCP, x: jnp.ndarray, u: jnp.ndarray):
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
        X_inv_M = jcp.linalg.solve(X_t, M_t)
        s_t = -jcp.linalg.solve(U_t - M_t.T @ X_inv_M, ru_t)
        r_t = -X_inv_M @ s_t
        return r_t, s_t

    r, s = jax.vmap(references)(Q, R, M, ru)
    lqt = LQT(
        A,
        B,
        jnp.zeros((T, nx)),
        Q[0],
        jnp.eye(nx),
        jnp.zeros(nx),
        Q,
        jnp.kron(jnp.ones((T, 1, 1)), jnp.eye(nx)),
        r,
        R,
        jnp.kron(jnp.ones((T, 1, 1)), jnp.eye(nu)),
        s,
        M,
    )
    return lqt


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


def par_Newton(
        nominal_states: jnp.ndarray,
        d: Derivatives,
        reg_param: float,
        ru: jnp.ndarray,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        M: jnp.ndarray,
):
    grad_cost_norm = jnp.linalg.norm(d.cu)
    reg_param = reg_param * grad_cost_norm
    R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), reg_param * jnp.eye(R.shape[1]))
    lqt = noc_to_lqt(ru, Q, R, M, d.fx, d.fu)
    Kx_par, d_par, S_par, v_par, pred_reduction, feasible = par_bwd_pass(lqt)
    du_par, dx_par = par_fwd_pass(
        lqt, jnp.zeros(nominal_states[0].shape[0]), Kx_par, d_par
    )
    return dx_par, du_par, pred_reduction, feasible, ru

def newton_oc(
        ocp: OCP,
        controls: jnp.ndarray,
        initial_state: jnp.ndarray,
        barrier_param: float,
):
    states = rollout(ocp.dynamics, controls, initial_state)
    initial_reg_param = 1.0
    initial_reg_inc = 2.0

    def while_body(val):
        x, u, iteration_counter, reg_param, reg_inc, _ = val

        # jax.debug.print("Iteration:    {x}", x=iteration_counter)

        cost = ocp.total_cost(x, u, barrier_param)
        # jax.debug.print("cost:         {x}", x=cost)

        derivatives = compute_derivatives(ocp, x, u, barrier_param)

        costates = par_costates(ocp, x[-1], derivatives)

        ru, Q, R, M = compute_lqr_params(costates, derivatives)

        def while_inner_loop(inner_val):
            _, _, _, _, rp, r_inc = inner_val
            dx, du, predicted_reduction, bwd_pass_feasible, Hu = par_Newton(
                x, derivatives, rp, ru, Q, R, M
            )
            temp_u = u + du
            temp_x = x + dx
            Hu_norm = jnp.max(jnp.abs(Hu))
            new_cost = jnp.where(
                check_traj_feasibility(ocp, temp_x, temp_u),
                ocp.total_cost(temp_x, temp_u, barrier_param),
                jnp.inf,
            )
            actual_reduction = new_cost - cost
            gain_ratio = actual_reduction / predicted_reduction
            succesful_minimzation = jnp.logical_and(gain_ratio > 0., bwd_pass_feasible)
            rp = jnp.where(
                succesful_minimzation,
                rp * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                rp * r_inc,
            )
            r_inc = jnp.where(succesful_minimzation, 2.0, 2 * r_inc)
            rp = jnp.clip(rp, 1e-16, 1e16)
            return temp_x, temp_u, succesful_minimzation, Hu_norm, rp, r_inc

        def while_inner_cond(inner_val):
            _, _, succesful_minimzation, _, _, _ = inner_val
            return jnp.logical_not(succesful_minimzation)

        x, u, _, Hamiltonian_norm, reg_param, reg_inc = lax.while_loop(while_inner_cond, while_inner_loop,
                                                              (x, u, jnp.bool_(0.), 0., reg_param, reg_inc))

        # jax.debug.print("a red:        {x}", x=actual_reduction)
        # jax.debug.print("p red         {x}", x=predicted_reduction)
        # jax.debug.print("|H_u|:        {x}", x=Hu_norm)

        iteration_counter = iteration_counter + 1
        # jax.debug.print("---------------------------------")
        # jax.debug.breakpoint()
        return x, u, iteration_counter, reg_param, reg_inc, Hamiltonian_norm

    def while_cond(val):
        _, _, iteration_counter, _, _, Hamiltonian_norm = val
        exit_cond = jnp.logical_or(
            Hamiltonian_norm < 1e-4, iteration_counter > 1000
        )
        return jnp.logical_not(exit_cond)

    (
        opt_x,
        opt_u,
        iterations,
        _,
        _,
        _,
    ) = lax.while_loop(
        while_cond,
        while_body,
        (
            states,
            controls,
            0,
            initial_reg_param,
            initial_reg_inc,
            jnp.array(1.0),
        ),
    )
    # jax.debug.print("Converged in {x} iterations", x=iterations)
    # jax.debug.breakpoint()
    return opt_x, opt_u, iterations


def par_interior_point_optimal_control(
        ocp: OCP,
        controls: jnp.ndarray,
        initial_state: jnp.ndarray,
):
    initial_barrier_param = 0.1

    def while_body(val):
        u, barrier_param, total_newton_iterations = val
        _, u, newton_iterations = newton_oc(
            ocp, u, initial_state, barrier_param
        )
        barrier_param = barrier_param / 5
        total_newton_iterations = total_newton_iterations + newton_iterations
        # jax.debug.breakpoint()
        return u, barrier_param, total_newton_iterations

    def while_cond(val):
        _, bp, _ = val
        return bp > 1e-4

    opt_u, _, N_iterations = lax.while_loop(
        while_cond, while_body, (controls, initial_barrier_param, 0)
    )
    # jax.debug.print("converged in {x}", x=t_conv)
    # opt_x = rollout(ocp.dynamics, opt_u, initial_state)
    # optimal_cost = ocp.total_cost(opt_x, opt_u, 0.0)
    # jax.debug.print("optimal cost {x}", x=optimal_cost)
    return opt_u, N_iterations
