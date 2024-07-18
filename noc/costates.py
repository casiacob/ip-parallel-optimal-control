import jax.numpy as jnp
from jax import lax, vmap, grad
from noc.optimal_control_problem import OCP, Derivatives


def combine_fc(elem1, elem2):
    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=False)


def par_init(F: jnp.ndarray, c: jnp.ndarray, x0: jnp.ndarray):
    tF0 = jnp.zeros_like(F[0])
    tc0 = F[0] @ x0 + c[0]

    def init(F_k, c_k):
        return F_k, c_k

    tF, tc = vmap(init)(F[1:], c[1:])

    tF = jnp.vstack((tF0.reshape(1, tF0.shape[0], tF0.shape[1]), tF))
    tc = jnp.vstack((tc0, tc))
    elems = (tF, tc)
    return elems

def par_costates(ocp: OCP, final_state: jnp.ndarray, d: Derivatives):
    lamda_T = grad(ocp.final_cost, 0)(final_state)
    F = jnp.transpose(d.fx[::-1, :, :], axes=(0, 2, 1))
    c = d.cx[::-1, :]
    elems = par_init(F, c, lamda_T)
    elems = par_scan(elems)
    return jnp.vstack((lamda_T, elems[1]))[::-1, :]


def seq_costates(ocp: OCP, final_state: jnp.ndarray, d: Derivatives):
    lamda_T = grad(ocp.final_cost, 0)(final_state)

    def body(carry, inp):
        lamda = carry
        cx, fx = inp
        lamda = cx + fx.T @ lamda
        return lamda, lamda

    _, lamda_opt = lax.scan(body, lamda_T, (d.cx, d.fx), reverse=True)
    lamda_opt = jnp.vstack((lamda_opt, lamda_T))
    return lamda_opt
