from typing import NamedTuple, Callable
import jax.numpy as jnp


class ocp(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
