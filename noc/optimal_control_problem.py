from typing import NamedTuple, Callable
import jax.numpy as jnp


class OCP(NamedTuple):
    dynamics: Callable
    constraints: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable


class Derivatives(NamedTuple):
    cx: jnp.ndarray
    cu: jnp.ndarray
    cxx: jnp.ndarray
    cuu: jnp.ndarray
    cxu: jnp.ndarray
    fx: jnp.ndarray
    fu: jnp.ndarray
    fxx: jnp.ndarray
    fuu: jnp.ndarray
    fxu: jnp.ndarray


class LinearizedOCP(NamedTuple):
    r: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    M: jnp.ndarray
