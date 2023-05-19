#
import warnings
from typing import List, Optional, Dict
import numpy as np

EPS = np.finfo(np.float32).eps
PROBLEM_TYPES = ["CVRP", "CCP"]


def eval_cop(solutions: List[Dict],
             k: Optional[int] = None,
             k_from_instance: bool = False,
             problem: Optional[str] = None,
             strict_feasibility: bool = True,
             **kwargs):
    """Wraps different evaluation schemes for several
    problems problems to ensure consistent evaluation."""
    results = []
    for sol in solutions:
        if problem is not None:
            if "problem" not in list(sol.keys()):
                sol['problem'] = problem
            else:
                assert sol['problem'].upper() == problem.upper()
        if sol['problem'].upper() == "CCP":
            res = eval_ccp(sol, k, k_from_instance, strict=strict_feasibility, **kwargs)
        elif sol['problem'].upper() in ["CVRP", "VRP"]:
            res = eval_cvrp(sol, k, k_from_instance, strict=strict_feasibility, **kwargs)
        else:
            raise ValueError(f"unknown problem: '{problem}'")
        results.append(res)

    infs = (
        np.array([r['tot_center_dist'] == float("inf") for r in results]) |
        np.array([r['nc'] == float("inf") for r in results])
    )
    num_inf = infs.sum()
    center_dist = [r['tot_center_dist'] for r in results if r['tot_center_dist'] != float("inf") ]
    nc = [r['nc'] for r in results if r['nc'] != float("inf")]

    summary = {
        "center_dist_mean": np.mean(center_dist) if len(center_dist) > 0 else float("inf"),
        "center_dist_std": np.std(center_dist) if len(center_dist) > 0 else float("inf"),
        "nc_mean": np.mean(nc) if len(nc) > 0 else float("inf"),
        "nc_std": np.std(nc) if len(nc) > 0 else float("inf"),
        "nc_median": np.median(nc) if len(nc) > 0 else float("inf"),
        "run_time_mean": np.mean([r['run_time'] for r in results]),
        "run_time_total": np.sum([r['run_time'] for r in results]),
        "num_infeasible": num_inf,
    }

    return results, summary


def eval_cvrp(
        solution: Dict,
        k: Optional[int] = None,
        k_from_instance: bool = False,
        strict: bool = True,
        **kwargs) -> Dict:
    """(Re-)Evaluate provided solutions for the CVRP."""
    DEPOT = 0

    data = solution['instance']
    coords = data.coords
    demands = data.demands
    routes = solution['assignment']
    k_max = data.num_components if k_from_instance else k

    # check feasibility of routes and calculate cost
    if routes is None or len(routes) == 0:
        k = float("inf")
        cost = float("inf")
    else:
        k = 0
        k_inf = False
        cost = 0.0
        for r in routes:
            if r and sum(r) > DEPOT:    # not empty and not just depot idx
                if r[0] != DEPOT:
                    r = [DEPOT] + r
                if r[-1] != DEPOT:
                    r.append(DEPOT)
                transit = 0
                source = r[0]
                cum_d = 0
                for target in r[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    cum_d += demands[target]
                    source = target
                if strict and cum_d > 1.0 + EPS:
                    print(f"capacity violation {(1 + EPS) - cum_d}: solution infeasible. "
                          f"Setting cost and k to 'inf'")
                    cost = float("inf")
                    k = float("inf")
                    break
                if strict and k_max is not None and k > k_max:
                    k_inf = True
                    cost = float("inf")
                cost += transit
                k += 1

        if k_inf:
            print(f"infeasible: k ({k}) > k_max ({k_max}). setting cost and k to 'inf'")
            k = float("inf")

    solution['nc'] = k
    solution['tot_center_dist'] = cost

    return solution


def eval_ccp(
        solution: Dict,
        k: Optional[int] = None,
        k_from_instance: bool = False,
        strict: bool = True,
        **kwargs) -> Dict:
    """(Re-)Evaluate provided solutions for the CVRP."""
    data = solution['instance']
    coords = data.coords
    demands = data.demands
    if k_from_instance:
        k = data.num_components

    assign = solution['assignment']
    if assign is None:
        sets = None
    else:
        sets = []
        n_unq = np.unique(assign)
        for lbl in n_unq:
            sets.append((assign==lbl).nonzero())

    # check feasibility of clusters and calculate cost
    if sets is None or len(sets) == 0:
        nc = float("inf")
        tot_dist = float("inf")
    else:
        nc = 0
        tot_dist = 0.0
        for s in sets:
            if s is not None and len(s) > 0:    # not empty
                center = coords[s].mean(0)
                cum_dist = np.linalg.norm(coords[s] - center, ord=2) ** 2   # squared euclidean distance
                cum_demand = demands[s].sum()
                if strict and cum_demand > 1.0 + EPS:
                    print(f"capacity violation {(1+EPS)-cum_demand}: solution infeasible. "
                          f"Setting cost and k to 'inf'")
                    tot_dist = float("inf")
                    nc = float("inf")
                    break
                tot_dist += cum_dist
                nc += 1

    if strict and k is not None:
        if nc > k:
            print(f"infeasible: nc ({nc}) > k_max ({k}). setting cost and k to 'inf'")
            tot_dist = float("inf")
            nc = float("inf")

    solution['nc'] = nc
    solution['tot_center_dist'] = tot_dist

    return solution

