#
import os
import warnings
from typing import Optional, Union, List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from lib.problems.generator import ProblemDataset, Generator
from lib.utils import CCPInstance
from baselines.CPP.methods_registry import rpack, ccp_mh


class ParallelSolver:
    """Parallelization wrapper for OR solvers
     based on multi-processing pool."""
    def __init__(self,
                 method: str = "rpack",
                 solver_args: Optional[Dict] = None,
                 num_workers: int = 1,
                 ):
        self.method = method.lower()
        self.solver_args = solver_args if solver_args is not None else {}
        if num_workers > os.cpu_count():
            warnings.warn(f"num_workers > num logical cores! This can lead to "
                          f"decrease in performance if env is not IO bound.")
        self.num_workers = num_workers

    @staticmethod
    def _solve(params: Tuple):
        """
        params:
            solver_cl: RoutingSolver.__class__
            data: GORTInstance
            solver_args: Dict
        """
        exe, instance, solver_args = params
        assignment, _ = exe(
            instance=instance,
            **solver_args
        )

        k = len(np.unique(assignment))
        instance = instance.update(labels=assignment, num_components=k)

        return instance

    def solve(self, data: List[CCPInstance]) -> List[CCPInstance]:

        assert isinstance(data[0], CCPInstance)
        if self.method == "rpack":
            exe = rpack
        elif self.method == "ccp_mh":
            exe = ccp_mh
        else:
            raise ValueError(self.method)

        if self.num_workers <= 1:
            results = list(tqdm(
                [self._solve((exe, d, self.solver_args)) for d in data],
                total=len(data)
            ))
        else:
            with Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(
                        self._solve,
                        [(exe, d, self.solver_args) for d in data]
                    ),
                    total=len(data),
                ))

            failed = [str(i) for i, res in enumerate(results) if res is None]
            if len(failed) > 0:
                warnings.warn(f"Some instances failed: {failed}")

        return results


def label_data(
    dataset: Union[ProblemDataset, List[CCPInstance]],
    save_path: str,
    method: str = "rpack",
    solver_kwargs: Optional[Dict] = None,
    num_workers: int = 1
):
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs
    solver = ParallelSolver(method, solver_kwargs, num_workers=num_workers)

    print(f"Solving {len(dataset)} instances...")
    instances = solver.solve(dataset)

    labeled_instances = [
        inst for inst in instances
        if inst.labels is not None and len(inst.labels) == inst.graph_size
    ]

    print(f"saving {len(labeled_instances)} labeled instances to {save_path}.")
    Generator.save_dataset(labeled_instances, filepath=save_path)


# ============= #
# ### TEST #### #
# ============= #
def _test():
    SIZE = 4
    N = 50
    SEED = 123

    ds = ProblemDataset(
        problem='ccp',
        coords_sampling_dist="gm",
        weights_sampling_dist="uniform",
        n_components=(2, 8),
        max_cap_factor=1.05,
        verbose=True
    )
    ds.seed(SEED)
    data = ds.sample(sample_size=SIZE, graph_size=N)

    label_data(
        method="ccp_mh",
        dataset=data,
        save_path="./data/_test/tst.npz",
        solver_kwargs={'num_init': 4, 'seed': SEED},
        num_workers=2
    )
