#
import os
import time
import warnings
import tqdm
from typing import List, Optional
from multiprocessing import Pool
from subprocess import check_call
import numpy as np

from lib.utils import CCPInstance


def write_instance(instance, instance_name, instance_filename, k: int = None):
    with open(instance_filename, "w") as f:
        n_nodes = len(instance[0]) - 1
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        if k is not None:
            f.write("VEHICLES : " + str(int(k)) + "\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : " + str(instance[2]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        for i in range(n_nodes + 1):
            f.write(" " + str(i + 1) + " " + str(instance[0][i][0] * s)[:15] + " " + str(instance[0][i][1] * s)[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2)+" "+str(instance[1][i])+"\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def write_para(dataset_name,
               instance_name,
               instance_filename,
               method,
               para_filename,
               max_trials=1000,
               time_limit=None,
               seed=1234,
               solution_filename=None):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("MTSP_MIN_SIZE = 0\n")
        f.write("SEED = " + str(seed) + "\n")
        if time_limit is not None:
            f.write("TIME_LIMIT = " + str(time_limit) + "\n")
        f.write("TRACE_LEVEL = 1\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"
        if solution_filename is not None:
            f.write(f"TOUR_FILE = {solution_filename}\n")   # to write best solution to file
        #f.write(f"OUTPUT_TOUR_FILE = {solution_filename}\n")


def read_results(log_filename, sol_filename):
    s = 1000000.0  # precision hardcoded by authors in write_instance()
    objs = []
    penalties = []
    runtimes = []
    running_objs, running_times = [], []
    num_vehicles = 0
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            if "VEHICLES" in line:
                l = line.strip().split(" ")
                num_vehicles = int(l[-1])
            elif line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]))
                penalties.append(int(line[-3]))
                runtimes.append(float(line[-1]))
            elif line[:1] == "*" and not line[:3] == "***":
                line = line.strip().split(" ")
                running_objs.append(int(line[-5][2:-1]) / s)
                running_times.append(float(line[-2]))

    tours = []
    dim, total_length = 0, 0
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):    # read out solution tours
            if "DIMENSION" in line:
                l = line.strip().split(" ")
                dim = int(l[-1])
            elif "Length" in line:
                l = line.strip().split(" ")
                total_length = int(l[-1])
            elif i > 5 and not "EOF" in line:
                idx = int(line)
                if i == 6:
                    assert idx == 1
                tours.append(idx)

    assert tours[-1] == -1
    assert len(tours) == dim + 1
    N = dim-num_vehicles

    # reformat tours
    tours = (np.array(tours) - 1).tolist()  # reduce idx by 1 (since TSPLIB format starts at 1)
    plan = []
    t = []
    for n in tours[1:]:
        if n <= 0 or n > N:
            plan.append(t)
            t = []
        else:
            t.append(n)
    assert len(plan) == num_vehicles

    # return objs, penalties, runtimes
    return {
        "objs": objs,
        "penalties": penalties,
        "runtimes": runtimes,
        "N": N,
        "num_vehicles": num_vehicles,
        "total_length": total_length,
        "solution": plan,
        "running_costs": running_objs,
        "running_times": running_times,
    }


def solve_LKH(dataset_name,
              instance,
              instance_name,
              rerun=False,
              max_trials=1000,
              time_limit=None,
              seed=1234,
              exe_path=None,
              k=None):
    para_filename = "result/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    solution_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".sol"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, k)
        write_para(dataset_name, instance_name, instance_filename,
                   "LKH", para_filename,
                   max_trials=max_trials,
                   time_limit=time_limit,
                   seed=seed,
                   solution_filename=solution_filename)
        with open(log_filename, "w") as f:
            check_call([str(exe_path), para_filename], stdout=f)
    return read_results(log_filename, solution_filename)


def run_lkh(args):
    return solve_LKH(*args)


def cvrp_inference(
        data: List[CCPInstance],
        lkh_exe_path: str,
        num_workers: int = 1,
        max_trials: int = 1000,
        time_limit: Optional[int] = None,
        seed: int = 1234,
        int_prec: int = 10000,
):
    method = "LKH"
    dataset_name = "CVRP"
    rerun = True
    if num_workers > os.cpu_count():
        warnings.warn(f"num_workers > num logical cores! This can lead to "
                      f"decrease in performance if env is not IO bound.")

    # set up directories
    os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)

    # convert data to input format for LKH
    # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
    dataset = [
        [
            d.coords.tolist(),
            np.ceil(d.demands * int_prec).astype(int).tolist(),
            int(d.constraint_value * int_prec)
        ] for d in data
    ]
    N = data[0].graph_size-1
    max_k = np.floor(np.max([np.sum(d.demands) for d in data]))
    K = int(max_k + 2 * np.ceil(np.sqrt(N)))

    if num_workers <= 1:
        results = list(tqdm.tqdm([
            solve_LKH(
                dataset_name,
                dataset[i],
                str(i),
                rerun,
                max_trials,
                time_limit,
                seed,
                lkh_exe_path,
                K
            )
            for i in range(len(dataset))
        ], total=len(dataset)))
    else:
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap(run_lkh, [
                (dataset_name, dataset[i], str(i), rerun, max_trials, time_limit, seed, lkh_exe_path, K)
                for i in range(len(dataset))
            ]), total=len(dataset)))

    s = 1000000.0   # precision hardcoded by authors in write_instance()
    objs = [np.array(r['objs'])/s for r in results]
    runtimes = [r['runtimes'] for r in results]

    solutions = [
        dict(
            solution=r['solution'],
            run_time=r['runtimes'][-1],
            problem=dataset_name.upper(),
            instance=d,
        ) for r, d in zip(results, data)
    ]


    results_ = {
        "objs": objs,
        "runtimes": runtimes,
    }
    return results_, solutions
