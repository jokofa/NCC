#
import os
from lib.problems.generator import ProblemDataset, Generator
from lib.problems.ccp_labeling import label_data

TYPE = "italia"     # "sample" # "shanghai"

if TYPE == "sample":
    SIZE = 4096     # size of dataset
    N_PARTS = 8
    CORES = 3
    METHOD = "ccp_mh"
    N_ITERS = 8

    K = (3, 12)
    N = 200             # number of nodes
    SEED = 1234
    MAX_CAP = 1.1   # [1.05, 1.1, 1.2, 1.5]
    COORDS_DIST = "gm"      # [gm, mixed]
    UNF_FRAC = 0.2  # fraction of uniformly sampled coords for mixed data

    k = K
    if isinstance(K, tuple):
        k = f"{K[0]}-{K[1]}"
        K = (K[0], K[1]+1)  # second value is exclusive in generator!
    dist = COORDS_DIST if COORDS_DIST == "gm" else f"{COORDS_DIST}_unf{UNF_FRAC}"

    save_pth = f"./data/CCP/CCP{N}/raw/"
    fname = f"train_{dist}_n{N}_k{k}_s{SIZE}_cap{str(MAX_CAP).replace('.', '_')}_seed{SEED}"

    assert SIZE % N_PARTS == 0
    part_size = SIZE//N_PARTS
    os.makedirs(save_pth, exist_ok=True)

    ds = ProblemDataset(
        problem="CCP",
        seed=SEED,
        coords_sampling_dist=COORDS_DIST,
        weights_sampling_dist="uniform",
        max_cap_factor=MAX_CAP,
        n_components=K,
        uniform_fraction=UNF_FRAC,
    )
    ds.seed(SEED)
    data = ds.sample(sample_size=SIZE, graph_size=N).data

    for i in range(N_PARTS):
        start_idx = i*part_size
        stop_idx = (i+1)*part_size
        d_slice = data[start_idx:stop_idx]
        # save as npz
        pth = os.path.join(save_pth, fname + f"_part{i}.npz")
        Generator.save_dataset(d_slice, filepath=pth)

    for i in range(N_PARTS):
        load_pth = os.path.join(save_pth, fname + f"_part{i}.npz")
        d_slice = Generator.load_dataset(load_pth)
        print(f"Labeling part {i+1}...")

        label_save_pth = os.path.join(save_pth, fname + f"_labeled_part{i}.npz")
        label_data(
            method=METHOD,
            dataset=d_slice,
            save_path=label_save_pth,
            solver_kwargs={'num_init': N_ITERS, 'seed': SEED,
                           't_local': 8, 't_total': 32},
            num_workers=CORES
        )

###
elif TYPE == "italia":

    DSET = "telecom_italia"
    P_SIZE = 512     # size of part
    N_PARTS = 10
    SIZE = N_PARTS*P_SIZE
    CORES = 2
    METHOD = "ccp_mh"
    N_ITERS = 10
    TLOC = 20
    TTOT = 200
    SEED = 1234
    N = 200
    CAP = 1.1

    save_pth = f"./data/CCP/benchmark/{DSET}/sub/raw/"
    fname = f"train_n{N}_s{SIZE}_cap{str(CAP).replace('.', '_')}_seed{SEED}"

    for i in range(6, N_PARTS):
        load_pth = os.path.join(save_pth, fname + f"_part{i}.npz")
        d_slice = Generator.load_dataset(load_pth)
        print(f"Labeling part {i+1}...")

        label_save_pth = os.path.join(save_pth, fname + f"_{METHOD}_labeled_part{i}.npz")
        label_data(
            method=METHOD,
            dataset=d_slice,
            save_path=label_save_pth,
            solver_kwargs={'num_init': N_ITERS, 'seed': SEED,
                           't_local': TLOC, 't_total': TTOT},
            num_workers=CORES
        )

###
elif TYPE == "shanghai":

    DSET = "shanghai_telecom"
    P_SIZE = 512  # size of part
    N_PARTS = 10
    SIZE = N_PARTS * P_SIZE
    CORES = 2
    METHOD = "ccp_mh"
    N_ITERS = 10
    TLOC = 20
    TTOT = 200
    SEED = 1234
    N = 200
    CAP = 1.1

    save_pth = f"./data/CCP/benchmark/{DSET}/sub/raw/"
    fname = f"train_n{N}_s{SIZE}_cap{str(CAP).replace('.', '_')}_seed{SEED}"

    for i in range(N_PARTS):
        load_pth = os.path.join(save_pth, fname + f"_part{i}.npz")
        d_slice = Generator.load_dataset(load_pth)
        print(f"Labeling part {i + 1}...")

        label_save_pth = os.path.join(save_pth, fname + f"_{METHOD}_labeled_part{i}.npz")
        label_data(
            method=METHOD,
            dataset=d_slice,
            save_path=label_save_pth,
            solver_kwargs={'num_init': N_ITERS, 'seed': SEED,
                           't_local': TLOC, 't_total': TTOT},
            num_workers=CORES
        )
