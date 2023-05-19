#
from timeit import default_timer
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.sweep import sweep_init
from verypy.util import sol2routes

from lib.utils.formats import CCPInstance


# routing_algo has
# - args: (distance matrix D, idx subset selected_idxs)
# - returns: (new_route, new_cost)

def solve_tsp_concorde(D, selected_idxs):
    raise NotImplementedError()


def eval_sweep_concorde(instance: CCPInstance, **kwargs):

    coords = instance.coords.copy()
    demands = instance.demands.copy()
    vehicle_capacity = instance.constraint_value

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = sweep_init(
        coordinates=coords,
        D=distances, d=demands, C=vehicle_capacity,
        direction="both", routing_algo=solve_tsp_concorde,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total



