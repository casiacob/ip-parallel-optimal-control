import jax.numpy as jnp
from jax import grad, jacfwd, lax, hessian
import jax
from noc.optimal_control_problem import CDerivatives, IPOCP
from noc.utils import rollout
from paroc import par_bwd_pass, par_fwd_pass
from paroc.lqt_problem import LQT


def compute_derivatives(ocp: IPOCP, states: jnp.ndarray, controls: jnp.ndarray):
    def body(x, u):
        cx_k = grad(ocp.stage_cost, 0)(x, u)
        cu_k = grad(ocp.stage_cost, 1)(x, u)

        cxx_k = hessian(ocp.stage_cost, 0)(x, u)
        cuu_k = hessian(ocp.stage_cost, 1)(x, u)
        cxu_k = jacfwd(jacfwd(ocp.stage_cost, 0), 1)(x, u)

        fx_k = jacfwd(ocp.dynamics, 0)(x, u)
        fu_k = jacfwd(ocp.dynamics, 1)(x, u)

        fxx_k = jacfwd(jacfwd(ocp.dynamics, 0), 0)(x, u)
        fuu_k = jacfwd(jacfwd(ocp.dynamics, 1), 1)(x, u)
        fxu_k = jacfwd(jacfwd(ocp.dynamics, 0), 1)(x, u)

        gx_k = jacfwd(ocp.constraints, 0)(x, u)
        gu_k = jacfwd(ocp.constraints, 1)(x, u)

        gxx_k = jacfwd(jacfwd(ocp.constraints, 0), 0)(x, u)
        guu_k = jacfwd(jacfwd(ocp.constraints, 1), 1)(x, u)
        gxu_k = jacfwd(jacfwd(ocp.constraints, 0), 1)(x, u)
        return (
            cx_k,
            cu_k,
            cxx_k,
            cuu_k,
            cxu_k,
            fx_k,
            fu_k,
            fxx_k,
            fuu_k,
            fxu_k,
            gx_k,
            gu_k,
            gxx_k,
            guu_k,
            gxu_k,
        )

    cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu, gx, gu, gxx, guu, gxu = jax.vmap(
        body
    )(states[:-1], controls)
    return CDerivatives(
        cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu, gx, gu, gxx, guu, gxu
    )


def compute_y_multipliers(
    ocp: IPOCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    z_multipliers: jnp.ndarray,
    cd: CDerivatives,
):
    xT = states[-1]
    y_T = grad(ocp.final_cost, 0)(xT)

    def body(carry, inp):
        y = carry
        x, u, z, cx, fx, gx = inp
        y = cx + fx.T @ y + gx.T @ z
        return y, y

    _, y_opt = lax.scan(
        body,
        y_T,
        (states[:-1], controls, z_multipliers, cd.cx, cd.fx, cd.gx),
        reverse=True,
    )
    y_opt = jnp.vstack((y_opt, y_T))
    return y_opt


def compute_lqr_params(
    ocp: IPOCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    z_multipliers: jnp.ndarray,
    slacks: jnp.ndarray,
    cd: CDerivatives,
    bp: float,
    rp: float,
):
    y_multipliers = compute_y_multipliers(ocp, states, controls, z_multipliers, cd)

    def body(
        x, u, y, z, s, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu, gx, gu, gxx, guu, gxu
    ):
        Hu = cu + fu.T @ y + gu.T @ z
        Hz = ocp.constraints(x, u) + s
        Hs = z - bp / s

        Q = cxx + jnp.tensordot(y.T, fxx, axes=1) + jnp.tensordot(z.T, gxx, axes=1)
        Huu = cuu + jnp.tensordot(y.T, fuu, axes=1) + jnp.tensordot(z.T, guu, axes=1)
        Huu = Huu + rp * jnp.eye(u.shape[0])
        Hxu = cxu + jnp.tensordot(y.T, fxu, axes=1) + jnp.tensordot(z.T, gxu, axes=1)

        Hss = jnp.diag(z / s)
        Huz = gu.T
        I = jnp.eye(z.shape[0])

        r = jnp.hstack((Hu, Hz, Hs))
        R = jnp.vstack(
            (
                jnp.hstack((Huu, Huz, jnp.zeros((u.shape[0], s.shape[0])))),
                jnp.hstack((Huz.T, jnp.zeros((z.shape[0], z.shape[0])), I)),
                jnp.hstack((jnp.zeros((s.shape[0], u.shape[0])), I, Hss)),
            )
        )
        M = jnp.vstack((Hxu.T, gx.T, jnp.zeros((x.shape[0], s.shape[0]))))
        A = fx
        B = jnp.hstack((fu, jnp.zeros((x.shape[0], 2 * z.shape[0]))))
        jax.debug.breakpoint()
        return r, Q, R, M.T, A, B, Hu, Huu

    return jax.vmap(body)(
        states[:-1],
        controls,
        y_multipliers[1:],
        z_multipliers,
        slacks,
        cd.cu,
        cd.cxx,
        cd.cuu,
        cd.cxu,
        cd.fx,
        cd.fu,
        cd.fxx,
        cd.fuu,
        cd.fxu,
        cd.gx,
        cd.gu,
        cd.gxx,
        cd.guu,
        cd.gxu,
    )


def noc_to_plqt(
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

    def references(Xt, Ut, Mt, rut):
        XiM = jnp.linalg.solve(Xt, Mt)
        st = -jnp.linalg.solve(Ut - Mt.T @ XiM, rut)
        rt = -XiM @ st
        return rt, st

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


def generate_projection_matrices(u: jnp.ndarray, z: jnp.ndarray, s: jnp.ndarray):
    nu = u.shape[1]
    nz = z.shape[1]
    ns = s.shape[1]
    Pu = jnp.hstack((jnp.eye(nu), jnp.zeros((nu, nz + ns))))
    Pz = jnp.hstack((jnp.zeros((nz, nu)), jnp.eye(nz), jnp.zeros((nz, ns))))
    Ps = jnp.hstack((jnp.zeros((ns, nu + nz)), jnp.eye(ns)))
    return Pu, Pz, Ps


def update_uzs(
    u: jnp.ndarray,
    z: jnp.ndarray,
    s: jnp.ndarray,
    augmented_du: jnp.ndarray,
    Pu: jnp.ndarray,
    Pz: jnp.ndarray,
    Ps: jnp.ndarray,
):

    def update_rule(uk, zk, sk, augmented_duk):
        duk = Pu @ augmented_duk
        dzk = Pz @ augmented_duk
        dsk = Ps @ augmented_duk

        uk = uk + duk

        max_z_step = jnp.where(-0.995 * zk / dzk > 0.0, -0.995 * zk / dzk, 0)
        max_z_step = jnp.min(max_z_step, initial=1.0, where=max_z_step > 0)
        zk = zk + max_z_step * dzk

        max_s_step = jnp.where(-0.995 * sk / dsk > 0.0, -0.995 * sk / dsk, 0)
        max_s_step = jnp.min(max_s_step, initial=1.0, where=max_s_step > 0)
        sk = sk + max_s_step * dsk

        return uk, zk, sk

    return jax.vmap(update_rule)(u, z, s, augmented_du)

def predicted_cost_reduction(ocp: IPOCP, cd: CDerivatives, Hu, Huu, x, u, z, s, k, bp):
    def body(Huk, Huuk, guk, xk, uk, zk, sk, kk):
        rp = ocp.constraints(xk, uk) + sk
        rd = zk*sk - bp
        rhat = zk * rp - rd
        Sigma = jnp.diag(zk/sk)
        Huk = Huk + guk.T @ jnp.diag(1/sk) @ rhat
        Huuk = Huuk + guk.T @ Sigma @ guk
        dVk = kk[0].T * Huk + 0.5 * kk[0].T * Huuk * kk[0]
        error = jnp.max(jnp.abs(jnp.hstack((Huk, rp, rd))))
        return dVk, error

    dV, err = jax.vmap(body)(Hu, Huu, cd.gu, x[:-1], u, z, s, k)
    return jnp.sum(dV), jnp.max(jnp.abs(err))


def par_solution(ocp: IPOCP, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, s: jnp.ndarray, bp: float, rp: float):
    Pu, Pz, Ps = generate_projection_matrices(u, z, s)
    cd = compute_derivatives(ocp, x, u)
    r, Q, R, M, A, B, Hu, Huu = compute_lqr_params(ocp, x, u, z, s, cd, bp, rp)
    I = jnp.eye(R.shape[1])
    R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), rp * I)
    lqt = noc_to_plqt(r, Q, R, M, A, B)
    Kx_par, d_par, S_par, v_par, pred_reduction, convex_problem = par_bwd_pass(lqt)
    _, error = predicted_cost_reduction(ocp, cd, Hu, Huu, x, u, z, s, d_par, bp)
    du_par, dx_par = par_fwd_pass(lqt, jnp.zeros(x[0].shape[0]), Kx_par, d_par)
    temp_x = x + dx_par
    temp_u, temp_z, temp_s = update_uzs(u, z, s, du_par, Pu, Pz, Ps)
    return temp_x, temp_u, temp_z, temp_s, pred_reduction, convex_problem, error

def ip_pd_oc(ocp: IPOCP, controls: jnp.ndarray, z_mult: jnp.ndarray, slacks: jnp.ndarray, initial_state: jnp.ndarray):
    rp0 = 10.
    nu0 = 2.
    states = rollout(ocp.dynamics, controls, initial_state)
    bp0 = ocp.total_cost(states, controls)/(z_mult.shape[0]*z_mult.shape[1])
    # bp0 = 10
    def while_body(val):
        x, u, z, s, t, bp, rp, nu, _, _ = val
        jax.debug.print("Iteration:    {x}", x=t)

        cost = ocp.barrier_total_cost(x, u, s, bp)
        jax.debug.print("cost:         {x}", x=cost)

        temp_x, temp_u, temp_z, temp_s, predicted_reduction, convex, Hu = par_solution(ocp, x, u, z, s, bp, rp)
        Hu_norm = jnp.max(jnp.abs(Hu))
        temp_s_feasible = jnp.all(temp_s > 0)
        temp_z_feasible = jnp.all(temp_z > 0)
        fp_feasible = jnp.logical_and(temp_s_feasible, temp_z_feasible)
        jax.debug.print("bwd f:        {x}", x=convex)
        jax.debug.print("fwd f:        {x}", x=fp_feasible)
        new_cost = ocp.barrier_total_cost(temp_x, temp_u, temp_s, bp)
        jax.debug.print("new cost:     {x}", x=new_cost)

        actual_reduction = new_cost - cost
        jax.debug.print("a red:        {x}", x=actual_reduction)
        jax.debug.print("p red         {x}", x=predicted_reduction)

        gain_ratio = actual_reduction / predicted_reduction
        jax.debug.print("gain ratio:   {x}", x=gain_ratio)

        accept_cond = jnp.logical_and(gain_ratio > 0, convex)
        accept_cond = jnp.logical_and(accept_cond, fp_feasible)

        rp = jnp.where(
            accept_cond,
            rp * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
            rp * nu,
        )
        nu = jnp.where(accept_cond, 2.0, 2 * nu)
        x = jnp.where(accept_cond, temp_x, x)
        u = jnp.where(accept_cond, temp_u, u)
        z = jnp.where(accept_cond, temp_z, z)
        s = jnp.where(accept_cond, temp_s, s)

        jax.debug.print("reg param:    {x}", x=rp)
        jax.debug.print("barrier  :    {x}", x=bp)

        t = t + 1
        jax.debug.print("|Hu|:         {x}", x=Hu_norm)
        bp = jnp.where(Hu_norm < bp, jnp.minimum(bp/5, bp**1.2), bp)
        jax.debug.print("---------------------------------")
        jax.debug.breakpoint()
        return x, u, z, s, t, bp, rp, nu, Hu_norm, convex

    def while_cond(val):
        _, _, _, _, t, bp, _, _, Hu_norm, bp_feasible = val
        # exit_cond = jnp.logical_and(Hu_norm < 1e-2, bp_feasible)
        exit_cond = t>200
        return jnp.logical_not(exit_cond)

    (
        opt_x,
        opt_u,
        opt_z,
        opt_s,
        _,
        opt_bp,
        _,
        _,
        _,
        _,
    ) = lax.while_loop(
        while_cond,
        while_body,
        (states, controls, z_mult, slacks, 0, bp0, rp0, nu0, jnp.array(1.0), jnp.bool(1.0)),
    )
    # jax.debug.breakpoint()
    return opt_x, opt_u