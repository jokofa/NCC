#
from typing import List, Union
import numpy as np
import torch

from lib.rl.env import CCPEnv
from lib.utils import CCPInstance, RPInstance


class RandomAgglomerative:
    """Randomized agglomerative clustering

    Args:
        problem: CCP or CVRP
        num_samples: number of random rollouts
        cuda: flag to run on GPU
        verbose: flag for increased verbosity
        render: flag to render
        use_nn_selection: instead of two random nodes, select first at random and
                            second as the nearest feasible neighbor of the first node
        **kwargs:
    """
    def __init__(self,
                 problem: str,
                 num_samples: int,
                 cuda: bool = False,
                 verbose: bool = False,
                 render: bool = False,
                 use_nn_selection: bool = False,
                 **kwargs):
        self.problem = problem.upper()
        self.num_samples = num_samples
        self.device = torch.device("cuda" if cuda else "cpu")
        self.verbose = verbose
        self.render = render
        self.use_nn_selection = use_nn_selection
        env_cl = CCPEnv if self.problem == "CCP" else None
        self.env = env_cl(
            check_feasibility=True,
            device=self.device,
            num_samples=self.num_samples,
            debug=self.verbose,
            enable_render=self.render,
            **kwargs
        )

    def seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.seed(seed+1)

    def assign(self, batch: Union[List[CCPInstance], List[RPInstance]]):
        if self.verbose:
            print(f"loading data...")
        self.env.load_data(batch)
        if self.verbose:
            print(f"starting assignment...")
        obs = self.env.reset()

        rews = []
        done = False
        i = 0
        while not done:
            d = obs.cluster_features.shape[1]
            msk = obs.node_i_mask
            node_i = []
            for bidx in range(self.env.bs):
                h = (d - msk[bidx].sum(-1))
                idx = np.random.choice(np.arange(h))
                node_i.append((msk[bidx] == 0).nonzero().view(-1)[idx].unsqueeze(-1))
            node_i = torch.cat(node_i, dim=-1)
            msk = self.env.get_node_j_mask(node_i)
            node_j = []
            for bidx in range(self.env.bs):
                if self.use_nn_selection:
                    coords = obs.cluster_features[bidx, :, 1:3]
                    node_i_idx = node_i[bidx]
                    node_i_coords = coords[node_i_idx]
                    candidate_idx = (msk[bidx] == 0).nonzero().view(-1)
                    candidate_coords = coords.gather(index=candidate_idx[:, None].expand(-1, 2), dim=0)
                    dist = torch.norm((candidate_coords-node_i_coords), p=2, dim=-1)
                    nearest_idx = dist.argmin()
                    node_j.append(candidate_idx[nearest_idx].unsqueeze(-1))
                else:
                    h = (d - msk[bidx].sum(-1))
                    idx = np.random.choice(np.arange(h))
                    node_j.append((msk[bidx] == 0).nonzero().view(-1)[idx].unsqueeze(-1))
            node_j = torch.cat(node_j, dim=-1)
            a = torch.cat((node_i[:, None], node_j[:, None]), dim=-1)
            obs, rew, done, info = self.env.step(a)
            rews.append(rew)

            if self.verbose:
                print(f"{i}:    a={a}\n-> rew={rew}")
                print(info)
            if self.render:
                self.env.render(as_gif=False)
            i += 1

        if self.verbose:
            print(f"total reward: {sum(rews)}")
        return self.env.export_sol()
