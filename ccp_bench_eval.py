
from copy import deepcopy
import os
import numpy as np

from lib.problems import ProblemDataset
from baselines.utils import eval_method
from baselines.CPP import methods_registry
from baselines.CPP.methods_registry import CUDA_METHODS
from lib.ltr.ccp.method import CKMeans
from lib.ltr.utils import load_model


SIZE = None
SEED = 1
NSEEDS = 1
CUDA = True
K_NOT_KNOWN = False
CORES = 4
T_LIM = 360     # 6min

SAVE_DIR = f"./outputs_eval/ccp_bench/"
BL_DIR = os.path.join(SAVE_DIR, "baselines")
M_DIR = os.path.join(SAVE_DIR, "model")
smp_cfg = {"sample_size": SIZE}
INF = float("inf")

DS_PTH = f"data/CCP/benchmark/stefanello/ccp_bench.npz"
CKPT = "outputs/final/shanghai_tel_ccp_200/gnn_pool_pointwise/2023-02-08_15-59-27_166847/checkpoints/epoch=189_val_acc=0.9791.ckpt"
NUM_INIT = 8
#MTHD = "ccp_mh"
#MTHD = "ncc_greedy"
MTHD = "ncc_samp"

result = None
metrics = {}
RESULTS = {}
seeds = [SEED+i for i in range(NSEEDS)]
ds = ProblemDataset(problem="CCP", seed=SEED, data_pth=DS_PTH)
ds = ds.sample(**smp_cfg, allow_pickle=True)

if MTHD == "ccp_mh":
    result, smry = eval_method(
        method=getattr(methods_registry, MTHD),
        dataset=ds,
        seeds=seeds,
        save_dir=BL_DIR,
        cuda=CUDA,
        k_not_known=K_NOT_KNOWN,
        sample_cfg=smp_cfg,
        num_init=NUM_INIT,
        num_cores=CORES,
        t_total=T_LIM,
        t_local=T_LIM//10,
        g_initial=10,
        l=10,
        init_method="capacity-based",
        raise_error=True
    )
    m_id = f"{MTHD}{'_cuda' if CUDA and MTHD in CUDA_METHODS else ''}"
    RESULTS[m_id] = result
    # replace infeasible runs with mean cost of random method
    res = deepcopy(result)
    costs = np.array([r['tot_center_dist'] for r in res])
    costs = costs.reshape(NSEEDS, -1)
    for i in range(len(costs)):
        inst_cost = costs[:, i]
        inf_msk = inst_cost == INF
        if np.any(inf_msk):
            print(f"inf: {inf_msk.sum()}")
            inst_cost[inf_msk] = 100
            costs[:, i] = inst_cost

    smry['center_dist_mean'] = np.mean(costs)
    smry['center_dist_std'] = np.mean(np.std(costs, axis=0))
    print(f"adapted summary: {smry}")
    metrics[m_id] = smry
    #print(RESULTS[m_id])

elif MTHD == "rpack":
    if not CUDA or CUDA and MTHD in CUDA_METHODS:
        result, smry = eval_method(
            method=getattr(methods_registry, MTHD),
            dataset=ds,
            seeds=seeds,
            save_dir=BL_DIR,
            cuda=CUDA,
            k_not_known=K_NOT_KNOWN,
            sample_cfg=smp_cfg,
            num_init=NUM_INIT,
            num_cores=CORES,
            gurobi_timeout=T_LIM//4,
            timeout=T_LIM,
            timeout_kill=(T_LIM*2)+1,
            verbose=False,
        )
        m_id = f"{MTHD}{'_cuda' if CUDA and MTHD in CUDA_METHODS else ''}"
        RESULTS[m_id] = result
        # replace infeasible runs with mean cost of random method
        res = deepcopy(result)
        costs = np.array([r['tot_center_dist'] for r in res])
        costs = costs.reshape(NSEEDS, -1)
        for i in range(len(costs)):
            inst_cost = costs[:, i]
            inf_msk = inst_cost == INF
            if np.any(inf_msk):
                print(f"inf: {inf_msk.sum()}")
                inst_cost[inf_msk] = 1000
                costs[:, i] = inst_cost

        smry['center_dist_mean'] = np.mean(costs)
        smry['center_dist_std'] = np.mean(np.std(costs, axis=0))
        print(f"adapted summary: {smry}")
        metrics[m_id] = smry

elif MTHD == "ncc_greedy":
    # greedily assigns the last 'opt_last_frac' fraction of total nodes
    # ordered by their absolute priority to the closest center
    model = load_model("ccp", CKPT)

    ckmeans = CKMeans(
        max_iter=50,
        num_init=NUM_INIT,
        model=model,
        seed=SEED,
        nbh_knn=25,
        init_method="ckm++",
        permute_k=False,
        tol=0.001,
        pre_iter=0,
        verbose=False,
        opt_last_frac=0.7,
        opt_last_samples=1,
        opt_last_prio=True
    )

    result, smry = eval_method(
        method=ckmeans.inference,
        dataset=ds,
        seeds=seeds,
        save_dir=M_DIR,
        cuda=CUDA,
        k_not_known=K_NOT_KNOWN,
        sample_cfg=smp_cfg,
        method_str=MTHD,
        time_limit=T_LIM,
    )
    m_id = f"{MTHD}{'_cuda' if CUDA and MTHD in CUDA_METHODS else ''}"
    RESULTS[m_id] = result
    # replace infeasible runs with mean cost of random method
    res = deepcopy(result)
    costs = np.array([r['tot_center_dist'] for r in res])
    costs = costs.reshape(NSEEDS, -1)
    for i in range(len(costs)):
        inst_cost = costs[:, i]
        inf_msk = inst_cost == INF
        if np.any(inf_msk):
            print(f"inf: {inf_msk.sum()}")
            inst_cost[inf_msk] = 100
            costs[:, i] = inst_cost

    smry['center_dist_mean'] = np.mean(costs)
    smry['center_dist_std'] = np.mean(np.std(costs, axis=0))
    print(f"adapted summary: {smry}")
    metrics[m_id] = smry
    #print(RESULTS[m_id])

elif MTHD == "ncc_samp":
    # samples multiple assignments for the last 'opt_last_frac' fraction of total nodes
    # and selects the best one
    model = load_model("ccp", CKPT)

    ckmeans = CKMeans(
        max_iter=50,
        num_init=NUM_INIT,
        model=model,
        seed=SEED,
        nbh_knn=25,
        init_method="ckm++",
        permute_k=False,
        tol=0.0001,
        pre_iter=0,
        verbose=False,
        opt_last_frac=0.7,
        opt_last_samples=64,
        opt_last_prio=True
    )

    result, smry = eval_method(
        method=ckmeans.inference,
        dataset=ds,
        seeds=seeds,
        save_dir=M_DIR,
        cuda=CUDA,
        k_not_known=K_NOT_KNOWN,
        sample_cfg=smp_cfg,
        method_str=MTHD,
    )
    m_id = f"{MTHD}{'_cuda' if CUDA and MTHD in CUDA_METHODS else ''}"
    RESULTS[m_id] = result
    # replace infeasible runs with mean cost of random method
    res = deepcopy(result)
    costs = np.array([r['tot_center_dist'] for r in res])
    costs = costs.reshape(NSEEDS, -1)
    for i in range(len(costs)):
        inst_cost = costs[:, i]
        inf_msk = inst_cost == INF
        if np.any(inf_msk):
            print(f"inf: {inf_msk.sum()}")
            inst_cost[inf_msk] = 100
            costs[:, i] = inst_cost

    smry['center_dist_mean'] = np.mean(costs)
    smry['center_dist_std'] = np.mean(np.std(costs, axis=0))
    print(f"adapted summary: {smry}")
    metrics[m_id] = smry
    #print(RESULTS[m_id])

else:
    raise RuntimeError


for i, r in zip(ds.data, result):
    print(i)
    print(r)

