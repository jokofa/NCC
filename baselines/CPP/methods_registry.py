#
from typing import Optional
from timeit import default_timer
from warnings import warn

import numpy as np
import torch
from torch_kmeans import ConstrainedKMeans
from torch_kmeans.clustering.constr_kmeans import InfeasibilityError

from lib.utils import CCPInstance
from lib.utils.utils import timeit
from lib.problems import ProblemDataset
from baselines.CPP.cap_knn import cap_knn
from baselines.CPP.GB21_MH.algorithm import gb21_mh
from baselines.CPP.rpack import rpack_exe
from baselines.CPP.rnd_agg import RandomAgglomerative

METHODS = [
    "random_select",
    "random_center_knn",
    "topk_center_knn",
    "cap_kmeans",
    "ccp_mh",
    "rpack",
    "agglomerative",
]

CUDA_METHODS = [
    "random_select",
    "random_center_knn",
    "topk_center_knn",
    "cap_kmeans",
    "agglomerative",
]


@timeit
def random_select(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        cuda: bool = False,
        verbose: bool = False,
        **kwargs
):
    """Select label for each node at random
    while trying to respect weight constraints."""
    rnd = np.random.default_rng(seed)
    n = instance.graph_size
    n_range = np.arange(n)

    if k is not None:
        # select k random centers
        idx = rnd.choice(n_range, size=k, replace=False)
        idx_msk = torch.ones((n,), dtype=torch.bool)
        idx_msk[idx] = 0
        assignment = cap_knn(
            coords=instance.coords,
            weights=instance.demands,
            center_idx=idx,
            node_idx_mask=idx_msk,
            cuda=cuda,
            verbose=verbose,
            random=True     # random assignment, not KNN!!
        )
    else:
        # find smallest k with feasible assignment
        assignment = None
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = int(np.floor(np.sum(instance.demands)))
        trials = 0
        max_trials = 2 * np.ceil(np.sqrt(n))
        while assignment is None:
            # select k random centers
            idx = rnd.choice(n_range, size=k, replace=False)
            idx_msk = torch.ones((n,), dtype=torch.bool)
            idx_msk[idx] = 0
            assignment = cap_knn(
                coords=instance.coords,
                weights=instance.demands,
                center_idx=idx,
                node_idx_mask=idx_msk,
                cuda=cuda,
                verbose=verbose,
                random=True
            )
            k += 1
            trials += 1
            if trials >= max_trials:
                break

    return assignment


@timeit
def random_center_knn(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        cuda: bool = False,
        verbose: bool = False,
        **kwargs
):
    """Capacitated KNN with random init - Select K random center nodes
    and sequentially add nodes in their neighborhood."""
    rnd = np.random.default_rng(seed)
    n = instance.graph_size
    n_range = np.arange(n)

    if k is not None:
        # select k random centers
        idx = rnd.choice(n_range, size=k, replace=False)
        idx_msk = torch.ones((n,), dtype=torch.bool)
        idx_msk[idx] = 0
        assignment = cap_knn(
            coords=instance.coords,
            weights=instance.demands,
            center_idx=idx,
            node_idx_mask=idx_msk,
            cuda=cuda,
            verbose=verbose,
        )
    else:
        # find smallest k with feasible assignment
        assignment = None
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = int(np.floor(np.sum(instance.demands)))
        trials = 0
        max_trials = 2 * np.ceil(np.sqrt(n))
        while assignment is None:
            # select k random centers
            idx = rnd.choice(n_range, size=k, replace=False)
            idx_msk = torch.ones((n,), dtype=torch.bool)
            idx_msk[idx] = 0
            assignment = cap_knn(
                coords=instance.coords,
                weights=instance.demands,
                center_idx=idx,
                node_idx_mask=idx_msk,
                cuda=cuda,
                verbose=verbose,
            )
            k += 1
            trials += 1
            if trials >= max_trials:
                break

    return assignment


@timeit
def topk_center_knn(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        cuda: bool = False,
        verbose: bool = False,
        **kwargs
):
    """Capacitated KNN with top-k init - Select K nodes with largest weights
    as centers and sequentially add nodes in their neighborhood."""

    n = instance.graph_size
    weights = instance.demands

    if k is not None:
        # select k nodes with largest weight
        if isinstance(weights, np.ndarray):
            w_idx = np.argsort(weights)[::-1].copy()    # need copy to get contiguous array
            idx = w_idx[:k]   # top-k selection
        else:
            idx = torch.topk(weights, k=k).indices
        idx_msk = torch.ones((n,), dtype=torch.bool)
        idx_msk[idx] = 0
        assignment = cap_knn(
            coords=instance.coords,
            weights=weights,
            center_idx=idx,
            node_idx_mask=idx_msk,
            cuda=cuda,
            verbose=verbose,
        )
    else:
        # find smallest k with feasible assignment
        assignment = None
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = int(np.floor(np.sum(instance.demands)))
        trials = 0
        max_trials = 2 * np.ceil(np.sqrt(n))
        while assignment is None:
            # select k nodes with largest weight
            if isinstance(weights, np.ndarray):
                w_idx = np.argsort(weights)[::-1].copy()  # need copy to get contiguous array
                idx = w_idx[:k]  # top-k selection
            else:
                idx = torch.topk(weights, k=k).indices
            idx_msk = torch.ones((n,), dtype=torch.bool)
            idx_msk[idx] = 0
            assignment = cap_knn(
                coords=instance.coords,
                weights=weights,
                center_idx=idx,
                node_idx_mask=idx_msk,
                cuda=cuda,
                verbose=verbose,
            )
            k += 1
            trials += 1
            if trials >= max_trials:
                break

    return assignment


def cap_kmeans(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        cuda: bool = False,
        num_init: int = 8,
        init_method: str = "rnd",
        n_parallel: int = 8,
        **kwargs
):
    """Capacitated kmeans clustering."""
    n = instance.graph_size
    coords = instance.coords
    weights = instance.demands

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    # cast to tensor
    coords = torch.from_numpy(coords) if isinstance(coords, np.ndarray) else coords
    weights = torch.from_numpy(weights) if isinstance(weights, np.ndarray) else weights
    coords = coords.unsqueeze(0).to(dtype=torch.float, device=device)
    weights = weights.unsqueeze(0).to(dtype=torch.float, device=device)

    num_init = 1 if init_method == "topk" else num_init

    if k is not None:
        model = ConstrainedKMeans(
            n_clusters=int(k),
            seed=seed,
            num_init=num_init,
            init_method=init_method,
            **kwargs
        )
        t_start = default_timer()
        try:
            assignment = model.fit_predict(x=coords, weights=weights)[0]
        except InfeasibilityError:
            assignment = None
        t_total = default_timer() - t_start
    else:
        org_max_iter = kwargs.pop("max_iter", None)
        model = ConstrainedKMeans(
            seed=seed,
            num_init=max(num_init//2, 1),
            init_method=init_method,
            max_iter=10,
            raise_infeasible=False,
            **kwargs
        )
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = int(np.floor(np.sum(instance.demands)))
        trials = 0
        n = instance.graph_size
        max_trials = 2 * np.ceil(np.sqrt(n))

        x = coords.repeat((n_parallel, 1, 1))
        w = weights.repeat((n_parallel, 1))
        try_k = torch.arange(k, k + n_parallel)

        t_start = default_timer()
        while True:
            try:
                assignment = model(x=x, weights=w, k=try_k).labels
            except InfeasibilityError:
                assignment = None
            if assignment is not None:
                if (assignment >= 0).all(dim=-1).any():
                    break
            else:
                try_k += n_parallel
                trials += n_parallel
            if trials >= max_trials:
                break

        # get the smallest k with a feasible assignment
        k_idx = (assignment < 0).any(dim=-1).float().argmin()
        k = int(try_k[k_idx])

        # run algorithm completely with original max iterations
        if org_max_iter is not None:
            kwargs['max_iter'] = org_max_iter
        model = ConstrainedKMeans(
            n_clusters=k,
            seed=seed,
            num_init=num_init,
            init_method=init_method,
            **kwargs
        )
        try:
            assignment = model(x=coords, weights=weights, k=k).labels
        except InfeasibilityError:
            assignment = None

        t_total = default_timer() - t_start

    return assignment.cpu().numpy() if assignment is not None else assignment, t_total


def ccp_mh(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_init: int = 8,
        t_total: int = 3600,
        t_local: int = 300,
        g_initial: int = 10,
        init_method: str = "kmeans++",
        l: int = 50,
        num_cores: int = 0,
        raise_error: bool = False,
        **kwargs
):
    """Run the meta-heuristic solver of Gnägi and Baumann.
    Default params are the recommended hyper-parameters from the paper.

    Args:
        seed:
        instance:
        k:
        num_init: number of runs of global optimization phase
        t_total: time limit on total running time
        t_local: time limit for solving model in local optimization phase
        g_initial: initial number of nearest medians to which an object can be assigned
        init_method: initialization method to determine initial set of medians
        l: number of nearest objects to each median to be considered as potential new medians

    Returns:

    """

    n = instance.graph_size
    coords = instance.coords
    weights = instance.demands

    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    Q = np.ones((n,))

    n_target = min(n, 200)  # target number of objects in initial subset
    _ = kwargs.pop("cuda", None)

    ###
    def run_(*args, n_start, total_time, local_time, **kwargs):
        try:
            med, assign = gb21_mh(
                *args,
                t_total=total_time,
                n_start=n_start,
                g_initial=g_initial,
                init=init_method,
                n_target=n_target,
                l=l,
                t_local=local_time,
                np_seed=seed,
                gurobi_seed=seed,
                cores=num_cores,
                **kwargs
            )
        except (RuntimeError, IndexError) as e:
            if raise_error:
                raise e
            warn(f"ccp_mh - ERROR: {e}")
            med, assign = None, None
        return med, assign

    t_start = default_timer()
    if k is not None:
        centers, assignment = run_(coords, Q, weights, int(k),
                                   n_start=num_init,
                                   total_time=t_total,
                                   local_time=t_local,
                                   **kwargs)
    else:
        # run for increasing k to find smallest k with feasible solution
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = np.floor(np.sum(weights))
        trials = 0
        max_trials = 2 * np.ceil(np.sqrt(n))
        while True:
            # run with small time limit and no local improvement routines
            _, assignment = run_(coords, Q, weights, int(k),
                                 n_start=2,
                                 total_time=8,
                                 local_time=0,
                                 no_local=True,
                                 **kwargs)
            if assignment is not None:
                #print(f"solution with k={k}")
                break
            else:
                k += 1
                trials += 1
            if trials >= max_trials:
                break
        centers, assignment = run_(coords, Q, weights, int(k),
                                   n_start=num_init,
                                   total_time=t_total,
                                   local_time=t_local,
                                   **kwargs)

    t_total = default_timer() - t_start

    # convert assignment
    if assignment is not None:
        unq = np.unique(assignment)
        unq_map = np.arange(k)
        for label, u in zip(unq_map, unq):
            assignment[assignment == u] = label

    return assignment, t_total


def rpack(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_init: int = 8,
        num_cores: int = 0,
        **kwargs
):
    """Run R-PACK solver of Lähderanta et al."""
    n = instance.graph_size
    coords = instance.coords
    weights = instance.demands
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    if k is not None:
        assignment, rt = rpack_exe(
            coords=coords,
            weights=weights,
            k=k,
            num_init=num_init,
            seed=seed,
            cores=num_cores,
            **kwargs,
        )
    else:
        print(f"no k provided. Searching for feasible k...")
        # a good proxy for the minimal k to start with is the min number
        # of clusters necessary to serve all weights under perfect allocation
        k = np.floor(np.sum(weights))
        trials = 0
        max_trials = 2 * np.ceil(np.sqrt(n))
        rts = []
        while True:
            assignment, rt = rpack_exe(
                coords=coords,
                weights=weights,
                k=k,
                num_init=min(4, num_init),
                verbose=1,
                seed=seed,
                cores=num_cores,
                gurobi_timeout=30,
                timeout=max(60, min(n//2, kwargs.get("timeout", 10e8)))
            )
            rts.append(rt)
            if assignment is not None:
                break
            else:
                k += 1
                trials += 1
            if trials >= max_trials:
                break

        _assignment = assignment
        assignment, rt = rpack_exe(
            coords=coords,
            weights=weights,
            k=k,
            num_init=num_init,
            seed=seed,
            cores=num_cores,
            **kwargs
        )
        rts.append(rt)
        rt = np.sum(rts)
        if assignment is None:
            assignment = _assignment

    return assignment, rt


def agglomerative(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        cuda: bool = False,
        num_init: int = 8,
        nn_selection: bool = False,
        **kwargs
):

    model = RandomAgglomerative(
        problem="CCP",
        cuda=cuda,
        num_samples=num_init,
        use_nn_selection=nn_selection,
        target_nc=int(k) if k is not None else None,
        **kwargs
    )
    model.seed(seed)

    t_start = default_timer()
    solutions = model.assign([instance])[0]
    t_total = default_timer() - t_start

    candidate_sol = solutions
    # get best sample
    if k is not None:
        correct_k = np.array([s['nc'] for s in solutions]) == k
        candidate_sol = [s for s, c in zip(solutions, correct_k) if c]
        if len(candidate_sol) == 0:
            candidate_sol = solutions
    min_cost_idx = np.argmin([s['tot_cost'] for s in candidate_sol])
    assignment = candidate_sol[min_cost_idx]['assignment']

    return assignment, t_total

