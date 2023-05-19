#
from typing import Optional
from timeit import default_timer
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.gapvrp import gap_init
from verypy.util import sol2routes

from lib.utils.formats import CCPInstance


def eval_gap(
        instance: CCPInstance,
        k: Optional[int] = None,
        min_k: bool = False,
        seed_method: str = "cones",
        try_all_seeds: bool = False,
        **kwargs):
    """
    Three phase cluster-first-route-second CVRP
    construction algorithm based on generalized assignment problem (GAP).
    The first two phases involve the clustering:
    First, a seed point is generated for each route,
    which is then used in approximating customer node service costs
    in solving a GAP relaxation of the VRP.
    The resulting assignments are then routed using a TSP solver.

    * seed_method="cones", the initialization method of Fisher and Jaikumar
        (1981) which can be described as Sweep with fractional distribution of
        customer demand and placing the seed points approximately to the center
        of demand mass of created sectors.
    * seed_method="kmeans", intialize seed points to k-means cluster centers.
    * seed_method="large_demands", according to Fisher and Jaikumar (1981)
        "Customers for which d_i > 1/2 C can also be made seed customers".
        However applying this rule relies on human operator who then decides
        the intuitively best seed points. This implementation selects the
        seed points satisfying the criteria d_i>mC, where m is the fractional
        capacity multipier, that are farthest from the depot and each other.
        The m is made iteratively smaller if there are no at least K seed point
        candidates.
    * seed_method="ends_of_thoroughfares", this option was descibed in
        (Fisher and Jaikumar 1981) as "Most distant customers at the end of
        thoroughfares leaving from the depot are natural seed customers". They
        relied on human operator. To automate this selection we make a
        DBSCAN clustering with eps = median 2. nearest neighbor of all nodes
        and min_samples of 3.

    """
    coords = instance.coords.copy()
    demands = instance.demands.copy()
    vehicle_capacity = instance.constraint_value

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = gap_init(
        points=coords.tolist(),
        D=distances, d=demands, C=vehicle_capacity, L=None, K=k,
        seed_method=seed_method,
        find_optimal_seeds=try_all_seeds and (seed_method in ["cones", "kmeans"]),
        minimize_K=min_k,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total

