#
from typing import Optional
from timeit import default_timer
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.petalvrp import petal_init
from verypy.util import sol2routes

from lib.utils.formats import CCPInstance


def eval_sweep_petal(
        instance: CCPInstance,
        k: Optional[int] = None,
        min_k: bool = False,
        **kwargs
):
    """Petal sets are created with sweep algorithm,
    then a set covering problem is solved with gurobi."""

    coords = instance.coords.copy()
    demands = instance.demands.copy()
    vehicle_capacity = instance.constraint_value

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = petal_init(
        points=coords.tolist(),
        D=distances, d=demands, C=vehicle_capacity, L=None, K=k,
        relaxe_SCP_solutions=False,     # no additional improvement heuristic moves
        minimize_K=min_k,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total

