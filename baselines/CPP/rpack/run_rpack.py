#
from typing import Optional, Union
from warnings import warn
import os
import tempfile
from timeit import default_timer
import subprocess
from inspect import cleandoc
import numpy as np
import pandas as pd


def filter_path(path: str):
    """
    Filter duplicates from environ path.
    Simplified from https://gist.github.com/elmotec/b61c65d54c581e16ab64
    """
    sep = os.pathsep
    elements = [e for e in path.split(sep)]
    seen = set()
    filtered_elements = [
        el for el in elements
        if el.upper() not in seen and not seen.add(el.upper())
    ]
    return sep.join(filtered_elements)


def rpack_exe(
    coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    seed: int = 42,
    num_init: int = 8,
    verbose: Union[bool, int] = True,
    gurobi_timeout: int = 60,
    timeout: int = 180,
    timeout_kill: Optional[int] = None,
    cores: int = 0,
    **kwargs
):
    """
    R-PACK solver of Lähderanta et al.
    https://github.com/terolahderanta/rpack
    """
    # create tmp dir
    with tempfile.TemporaryDirectory() as TMP:
        fpath = os.path.join(TMP, "instance.csv")
        exe_path = os.path.dirname(os.path.abspath(__file__))

        # save instance as csv
        idx = np.zeros(coords.shape[0])
        dat = np.concatenate([idx[:, None], coords, weights[:, None]], axis=-1)
        df = pd.DataFrame(dat, columns=["batch_id", "x_coord", "y_coord", "weight"])
        df.to_csv(fpath, index=False)

        # clean up environ
        os.environ["PATH"] = filter_path(os.environ["PATH"])
        os.environ["LD_LIBRARY_PATH"] = filter_path(os.environ["LD_LIBRARY_PATH"])
        # run rpack
        # set ENV variables for Gurobi
        GHOME = "/opt/gurobi912/linux64"
        os.environ["GUROBI_HOME"] = GHOME
        if f"{GHOME}/bin" not in os.environ["PATH"]:
            os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['GUROBI_HOME']}/bin"
        if f"{GHOME}/lib" not in os.environ["LD_LIBRARY_PATH"]:
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['GUROBI_HOME']}/lib"
        # create bash script
        script = cleandoc(F"""
            #!/bin/sh
            Rscript --vanilla {exe_path}/run.R -f {fpath} --path {TMP} -l {num_init} -k {int(k)} --seed {seed} -c {cores} -t --timeout {timeout} --gurobi_timeout {gurobi_timeout} {'-v' if verbose else ""}
            wait
            """)
        #print(len(script), script)
        t_start = default_timer()
        try:
            if int(verbose) > 1:
                p = subprocess.run(script, shell=True,
                                   executable='/bin/bash',
                                   timeout=timeout_kill)
            else:
                p = subprocess.run(script, shell=True,
                                   executable='/bin/bash',
                                   timeout=timeout_kill,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            rt = default_timer() - t_start
            warn(f"{__name__}: Timeout expired. ({rt}s)")
            assignment = None
            return assignment, rt

        t_total = default_timer() - t_start
        # read solution and runtime
        res_pth = os.path.join(TMP, "rpack_results.csv")
        if os.path.exists(res_pth):
            result = pd.read_csv(res_pth)
            assert len(result) == len(coords)
            rt_pth = os.path.join(TMP, "rpack_runtime.csv")
            rt = pd.read_csv(rt_pth)
            # get assignment
            assignment = result['rpack_label'].to_numpy()
            rt = rt['time_table'].to_numpy()[0]
            if verbose:
                print(f"feasible assignment was found! k={k}")
        else:
            # no feasible assignment
            if verbose:
                print(f"no feasible assignment found. k={k}")
            assignment = None
            rt = t_total

    return assignment, rt
