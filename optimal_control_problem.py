from typing import NamedTuple, Callable


class ocp(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
