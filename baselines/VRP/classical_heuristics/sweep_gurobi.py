#
from timeit import default_timer
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.sweep import sweep_init
from verypy.util import sol2routes
from verypy.tsp_solvers.tsp_solver_gurobi import solve_tsp_gurobi

from lib.utils.formats import CCPInstance


def eval_sweep_gurobi(
        instance: CCPInstance,
        min_k: bool = False,
        **kwargs
):

    coords = instance.coords.copy()
    demands = instance.demands.copy()
    vehicle_capacity = instance.constraint_value

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = sweep_init(
        coordinates=coords,
        D=distances, d=demands, C=vehicle_capacity,
        direction="both", routing_algo=solve_tsp_gurobi,
        minimize_K=min_k,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total



