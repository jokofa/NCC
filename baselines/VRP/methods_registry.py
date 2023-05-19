#
import os
from warnings import warn
from typing import Optional
from timeit import default_timer
import numpy as np
import torch
from gurobipy import GurobiError

from lib.utils import CCPInstance
from lib.problems import ProblemDataset
# POMO
import baselines.VRP.POMO.POMO.cvrp.source.MODEL__Actor.grouped_actors as POMOActor
from baselines.VRP.POMO.pomo import eval_model as eval_pomo

METHODS = [
    "savings",
    "sweep",
    "sweep_gurobi",
    "sweep_petal",
    "gap",
    "pomo",
]

CUDA_METHODS = [
    "pomo",
]

TLIM = 600


def runtime_param_setter(num_cores: int = 1, time_limit: int = None):
    # way of changing config parameters
    # which were hardcoded in verypy config.py
    t_lim = time_limit if time_limit is not None else TLIM
    import verypy.config as cfg
    if cfg.MIP_SOLVER_THREADS != num_cores:
        cfg.MIP_SOLVER_THREADS = num_cores
    cfg.MAX_MIP_SOLVER_RUNTIME = t_lim
    runtime_param_checker(num_cores)


def runtime_param_checker(num_cores):
    from verypy.classic_heuristics.petalvrp import MIP_SOLVER_THREADS as mst
    assert mst == num_cores


def savings(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_cores: int = 1,
        savings_function: str = "clarke_wright",
        **kwargs
):
    """Savings construction heuristic
    with either savings function:

        - 'clarke_wright'
        - 'gaskell_lambda'
        - 'gaskell_pi'

    """

    runtime_param_setter(num_cores=num_cores)
    from baselines.VRP.classical_heuristics.savings import eval_savings

    np.random.seed(seed)
    if k is not None:
        # cannot solve with specified k, but can minimize k
        warn(f"savings cannot solve directly for specified k, "
             f"but will try to minimize k.")
        kwargs.pop("min_k", False)
        kwargs['min_k'] = True

    solution, rt = eval_savings(
        instance=instance,
        savings_function=savings_function,
        **kwargs
    )

    return solution, rt


def sweep(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_cores: int = 1,
        **kwargs
):
    """Sweep construction heuristic,
    nodes are routed in order of assignment."""

    runtime_param_setter(num_cores=num_cores)
    from baselines.VRP.classical_heuristics.sweep import eval_sweep

    np.random.seed(seed)
    if k is not None:
        # cannot solve with specified k, but can minimize k
        warn(f"sweep cannot solve directly for specified k, "
             f"but will try to minimize k.")
        kwargs.pop("min_k", False)
        kwargs['min_k'] = True

    solution, rt = eval_sweep(
        instance=instance,
        **kwargs
    )

    return solution, rt


def sweep_gurobi(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_cores: int = 1,
        t_lim: Optional[int] = None,
        **kwargs
):
    """Sweep construction heuristic,
    nodes are routed according to MIP solution by gurobi."""

    runtime_param_setter(num_cores=num_cores, time_limit=t_lim)
    from baselines.VRP.classical_heuristics.sweep_gurobi import eval_sweep_gurobi

    np.random.seed(seed)
    if k is not None:
        # cannot solve with specified k, but can minimize k
        warn(f"sweep (gurobi) cannot solve directly for specified k, "
             f"but will try to minimize k.")
        kwargs.pop("min_k", False)
        kwargs['min_k'] = True

    solution, rt = eval_sweep_gurobi(
        instance=instance,
        **kwargs
    )

    return solution, rt


def sweep_petal(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_cores: int = 1,
        t_lim: Optional[int] = None,
        **kwargs
):
    """Petal sets are created with sweep algorithm,
    then a set covering problem is solved with gurobi."""

    runtime_param_setter(num_cores=num_cores, time_limit=t_lim)
    from baselines.VRP.classical_heuristics.sweep_petal import eval_sweep_petal

    np.random.seed(seed)
    solution, rt = eval_sweep_petal(
        instance=instance,
        k=k,
        **kwargs
    )

    return solution, rt


def gap(
        seed: int,
        instance: CCPInstance,
        k: Optional[int] = None,
        num_cores: int = 1,
        seed_method: str = "cones",
        try_all_seeds: bool = False,
        t_lim: Optional[int] = None,
        **kwargs
):
    """Three phase cluster-first-route-second CVRP
    construction algorithm based on generalized assignment problem (GAP)."""

    runtime_param_setter(num_cores=num_cores, time_limit=10 if try_all_seeds else t_lim)
    from baselines.VRP.classical_heuristics.gap import eval_gap

    np.random.seed(seed)
    t_start = default_timer()
    try:
        solution, rt = eval_gap(
            instance=instance,
            k=k,
            seed_method=seed_method,
            try_all_seeds=try_all_seeds,
            **kwargs
        )
    except (UnboundLocalError, GurobiError):
        solution = None
        rt = default_timer() - t_start

    return solution, rt


def pomo(
    instance: CCPInstance,
    k: Optional[int] = None,
    cuda: bool = False,
    ckpt_pth: str = "baselines/VRP/POMO/POMO/cvrp/result/Saved_CVRP100_Model/ACTOR_state_dic.pt",
    augment: bool = True,  # flag to use 8-fold instance data augmentation
    single_trajectory: bool = False,
    **kwargs
):
    assert ckpt_pth is not None and os.path.exists(ckpt_pth)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    rollout_cfg = {
        "augment": augment,
        "single_trajectory": single_trajectory
    }

    policy = POMOActor.ACTOR().to(device)
    policy.load_state_dict(torch.load(ckpt_pth))
    policy.eval()

    if k is not None:
        warn(f"POMO cannot solve directly for specified k, "
             f"it will only optimize k in terms of optimal cost.")

    solutions, rtimes = eval_pomo(
        data=[instance],
        actor=policy,
        device=device,
        batch_size=1,
        rollout_cfg=rollout_cfg,
    )

    return solutions[0], rtimes[0]
