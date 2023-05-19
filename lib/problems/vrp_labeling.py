#
from typing import Optional, List, Dict
import numpy as np

from lib.problems.generator import ProblemDataset, Generator
from lib.utils import CCPInstance
from baselines.VRP.LKH.lkh import cvrp_inference

EXE_PTH = "./baselines/VRP/LKH/LKH"


def label_data(
    dataset: List[CCPInstance],
    save_path: str,
    solver_kwargs: Optional[Dict] = None,
    num_workers: int = 1,
    seed: int = 1234,
    lkh_exe_path: str = EXE_PTH,
):
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs
    print(f"Solving {len(dataset)} instances...")
    _, solutions = cvrp_inference(
        data=dataset,
        lkh_exe_path=lkh_exe_path,
        num_workers=num_workers,
        seed=seed,
        **solver_kwargs
    )
    print(f"done.")
    labeled_instances = []
    for solution in solutions:
        sol = solution['solution']
        inst = solution['instance']
        # parse solution
        n = inst.graph_size
        assignment = np.ones(n-1)
        cl = 0
        for t in sol:
            if len(t) > 0:
                tr = np.array(t)
                assert np.all(tr > 0), f"depot cannot be part of solution!"
                assignment[tr-1] = cl
                cl += 1

        # sanity check
        d_total = inst.demands.sum()
        k_used = len(np.unique(assignment))
        assert k_used >= d_total
        labeled_instances.append(inst.update(labels=assignment, num_components=k_used))

    print(f"saving {len(labeled_instances)} labeled instances to {save_path}.")
    Generator.save_dataset(labeled_instances, filepath=save_path)


# ============= #
# ### TEST #### #
# ============= #
def _test():
    SIZE = 5
    N = 100
    SEED = 123
    SPTH = "./data/_test/tst.npz"
    problem = 'cvrp'
    coord_samp = 'mixed'  # ['uniform', 'gm', 'mixed']
    weight_samp = 'random_k_variant'  # ['random_int', 'uniform', 'gamma', 'random_k_variant']
    k = 6
    cap = 30
    max_cap_factor = 1.1

    ds = ProblemDataset(
        problem=problem,
        seed=SEED,
        coords_sampling_dist=coord_samp,
        weights_sampling_dist=weight_samp,
        n_components=3,
    )
    data = ds.sample(
        sample_size=SIZE,
        graph_size=N,
        k=k,
        cap=cap,
        max_cap_factor=max_cap_factor
    )

    label_data(
        dataset=data,
        save_path=SPTH,
        solver_kwargs={'max_trials': 100},
        seed=SEED,
        num_workers=2
    )

    instances = Generator.load_dataset(filename=SPTH)
    print(instances)
    i0 = instances[0]
    print(i0.labels)
    print(i0.graph_size, len(i0.labels))
