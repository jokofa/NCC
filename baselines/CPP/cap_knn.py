#
from typing import Tuple, Union
import numpy as np
import torch


def cap_knn(
    coords: Union[torch.Tensor, np.ndarray],
    weights: Union[torch.Tensor, np.ndarray],
    center_idx: Union[torch.Tensor, np.ndarray],
    node_idx_mask: Union[torch.Tensor, np.ndarray],
    cuda: bool = False,
    random: bool = False,
    verbose: bool = True,
):
    """Sequentially add nodes in neighborhood of each provided center
    if it has remaining capacity until all nodes are served."""

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    # to tensor
    coords = torch.from_numpy(coords) if isinstance(coords, np.ndarray) else coords.clone()
    weights = torch.from_numpy(weights) if isinstance(weights, np.ndarray) else weights.clone()
    center_idx = torch.from_numpy(center_idx) if isinstance(center_idx, np.ndarray) else center_idx.clone()
    # to dtype/device
    coords = coords.to(dtype=torch.float, device=device)
    weights = weights.to(dtype=torch.float, device=device)
    center_idx = center_idx.to(dtype=torch.long, device=device)
    node_idx_mask = node_idx_mask.to(device=device).clone()
    n_org = coords.size(0)

    # select centers from coords and weights
    centers = coords[center_idx]
    center_weights = weights[center_idx]
    coords = coords[node_idx_mask]
    weights = weights[node_idx_mask]

    k = len(center_idx)
    n = coords.size(0)
    if random:
        indices = torch.multinomial(
            torch.ones((k, n), device=device),
            n, replacement=False
        )
    else:
        # pairwise distances from each center to each node
        dist_mat = torch.cdist(centers, coords, p=2) ** 2
        values, indices = dist_mat.sort(dim=-1)

    # prepare buffers and masks
    assignment = -torch.ones(n, dtype=torch.long, device=device)
    remaining_cap = torch.ones(k, dtype=torch.float, device=device) - center_weights
    available_indices = torch.ones_like(indices).bool()

    assignment_changed = True
    while assignment_changed:
        assignment_old = assignment.clone()
        # sequentially add one node to each center
        # if it does not violate the remaining capacity
        # and mask its index for all other centers
        for i in range(k):
            candidates = indices[available_indices].view(k, -1)[i]
            for j in range(candidates.size(0)):
                candidate = candidates[j]
                weight = weights[candidate]
                if weight <= remaining_cap[i]:
                    assignment[candidate] = i   # assign center label
                    remaining_cap[i] -= weight  # adapt remaining capacity
                    available_indices[indices == candidate] = 0     # mask for all others
                    break
        assignment_changed = (assignment_old != assignment).any()

    if (assignment < 0).any():
        if verbose:
            print(f"k={k}: there were unassigned nodes!")
        return None

    labels = torch.empty(n_org, dtype=torch.long, device=device)
    labels[node_idx_mask] = assignment
    labels[center_idx] = torch.arange(k, device=device)   # reassign centers

    return labels.cpu().numpy()
