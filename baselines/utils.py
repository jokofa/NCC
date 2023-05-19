#
import os
from typing import Optional, Union, Dict, List, Callable
from copy import deepcopy
from tqdm import tqdm
import random
import numpy as np
import torch

from lib.problems import eval_cop, ProblemDataset
from lib.utils.runner_utils import save_results


def eval_method(
        method: Callable,
        dataset: Union[ProblemDataset, str],
        problem: str = "ccp",
        seeds: Union[int, List[int]] = 1234,
        save_dir: str = "./outputs_eval/baselines/",
        cuda: bool = False,
        k_not_known: bool = False,
        strict_feasibility: bool = True,
        disable_progress_bar: bool = False,
        sample_cfg: Optional[Dict] = None,
        method_str: Optional[str] = None,
        **kwargs
):
    method_str = method.__name__ if method_str is None else method_str
    print(f"evaluating baseline: '{method_str}'")

    problem = problem.lower()
    # setup dataset
    if isinstance(dataset, ProblemDataset):
        ds_str = os.path.basename(dataset.data_pth) if dataset.data_pth is not None else ""
        assert dataset.data is not None and len(dataset.data) > 0
        test_ds = dataset
    else:
        print(f"Dataset path provided. Loading dataset...")
        assert isinstance(dataset, str) and os.path.exists(dataset)
        ds_str = os.path.basename(dataset)
        sample_cfg = sample_cfg if sample_cfg is not None else {}
        test_ds = ProblemDataset(
            problem=problem,
            data_pth=dataset,
        ).sample(**sample_cfg)

    graph_size = test_ds.data[0].graph_size

    p_str = f"{test_ds.problem.lower()}{graph_size}"
    print(f"loaded test data: {p_str} -> {ds_str}")

    save_dir = os.path.join(save_dir, method_str, os.path.splitext(ds_str)[0])
    os.makedirs(save_dir, exist_ok=True)
    seeds = seeds if isinstance(seeds, list) else [seeds]

    print(f"running method for {len(seeds)} seeds on {len(test_ds)} instances...")
    solutions = []
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        test_ds.seed(sd)
        print(f"Running eval for seed {sd}...")

        sols = []
        for inst in tqdm(test_ds, disable=disable_progress_bar):
            k = None if k_not_known else inst.num_components
            assign, rt = method(
                seed=sd,
                instance=inst,
                k=k,
                cuda=cuda,
                **kwargs
            )

            sols.append({
                "instance": inst,
                "assignment": assign,
                "run_time": rt,
            })

        solutions += deepcopy(sols)
        eval_results, _ = eval_cop(
            solutions=sols,
            k_from_instance=(not k_not_known),
            problem=problem,
            strict_feasibility=strict_feasibility,
        )
        save_results(eval_results, save_dir=save_dir, postfix=f"_seed{sd}")

    print("-----------------------------------------")
    results, summary = eval_cop(
        solutions=solutions,
        k_from_instance=(not k_not_known),
        problem=problem,
        strict_feasibility=strict_feasibility,
    )
    #print(f"summary: {summary}")
    save_results(results, save_dir=save_dir, postfix=f"_full_{len(seeds)}")

    return results, summary

