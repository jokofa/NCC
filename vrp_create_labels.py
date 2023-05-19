#
from argparse import ArgumentParser
import os
from lib.problems.generator import ProblemDataset, Generator
from lib.problems.vrp_labeling import label_data

parser = ArgumentParser(description='Run experiments on ISMLL SLURM cluster')
parser.add_argument('--cores', '-c', type=int, default=3)
args = parser.parse_args()


P_SIZE = 512     # size of part
N_PARTS = 10
SIZE = N_PARTS*P_SIZE
CORES = args.cores
print(f"running on {CORES} cores.")

N_TRIALS = 10000
TIMEOUT = 600
SEED = 1234
N = 200
CAP = 1.1
K = 30
coord_samp = 'mixed'  # ['uniform', 'gm', 'mixed']
unf_frac = 0.2
weight_samp = 'random_k_variant'  # ['random_int', 'uniform', 'gamma', 'random_k_variant']
dist = coord_samp if coord_samp == "gm" else f"{coord_samp}_unf{unf_frac}"

save_pth = f"./data/VRP/VRP{N}/raw/"
fname = f"train_{dist}_n{N}_kmax{K}_s{SIZE}_cap{str(CAP).replace('.', '_')}_seed{SEED}"

cap = 50

ds = ProblemDataset(
    problem='cvrp',
    seed=SEED,
    coords_sampling_dist=coord_samp,
    weights_sampling_dist=weight_samp,
    n_components=3,
    uniform_fraction=unf_frac,
)
data = ds.sample(
    sample_size=SIZE,
    graph_size=N,
    k=K,
    cap=cap,
    max_cap_factor=CAP
)

assert SIZE % N_PARTS == 0
part_size = SIZE//N_PARTS
os.makedirs(save_pth, exist_ok=True)

for i in range(N_PARTS):
    start_idx = i*part_size
    stop_idx = (i+1)*part_size
    d_slice = data[start_idx:stop_idx]
    # save as npz
    pth = os.path.join(save_pth, fname + f"_part{i}.npz")
    Generator.save_dataset(d_slice, filepath=pth, problem="vrp")

for i in range(N_PARTS):
    load_pth = os.path.join(save_pth, fname + f"_part{i}.npz")
    d_slice = Generator.load_dataset(load_pth)
    print(f"Labeling part {i+1}...")

    label_save_pth = os.path.join(save_pth, fname + f"_labeled_part{i}.npz")
    label_data(
        dataset=d_slice,
        save_path=label_save_pth,
        solver_kwargs={'max_trials': N_TRIALS, 'time_limit': TIMEOUT},
        seed=SEED,
        num_workers=CORES
    )

