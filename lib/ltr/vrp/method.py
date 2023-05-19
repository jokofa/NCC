#
from typing import Union, Any, Optional, Tuple
from warnings import warn
from timeit import default_timer
from multiprocessing import Pool
import psutil
import numpy as np
from scipy.spatial import distance_matrix as calc_distance_matrix
import torch
from torch import LongTensor, Tensor
from torch_kmeans.utils.distances import BaseDistance, LpDistance
from torch_kmeans.utils.utils import group_by_label_mean
from torch_kmeans.clustering.constr_kmeans import InfeasibilityError
#from gurobipy import GurobiError
from verypy.tsp_solvers.tsp_solver_fast import solve_tsp_fast

from lib.utils import CCPInstance
from lib.ltr.ccp.method import CKMeans
from lib.ltr.vrp.vrp_model import VRPModel
from lib.ltr.utils import knn_graph, CycleIdx


class RouteCKMeans(CKMeans):
    """
    Constrained k-means as cluster-first-route-second (CFRS).
    CKMeans solving the tour assignment for
    capacitated routing problems and then solving the
    resulting TSPs with a dedicated TSP solver.

    Args:
        model: learned score function estimator
        depot_priority: add distance to depot in priority calculation,
                        the idea is that nodes far away from the depot are hard
                        to reassign to other tours and thus should be
                        assigned early, i.e. with higher priority
                        (default: True)
        depot_priority_add: if True, add to priority scaling factor, else
                            multiply priority by distance to depot
        ########
        normalize: String id of method to use to normalize input.
                    one of ['mean', 'minmax', 'unit'].
                    None to disable normalization. (default: None).
        priority_scaling: method to use for priority scaling
        vanilla_priority: only use heuristic priorities without score function
        nbh_knn: size of KNN graph neighborhood to consider in model (default: 16)
        convergence_criterion: criterion to check for convergence
                                one of ['shift', 'inertia'], (default: 'inertia')
        opt_last_frac: fraction of remaining points to be greedily assigned
                        at the end of the assignment step
        opt_last_samples: how many rollouts to sample for opt_last_frac
        opt_last_prio: if optimization of last assignments should be done
                        according to priorities, otherwise only use weights
        ########
        init_method: Method to initialize cluster centers:
                        ['rnd', 'topk', 'k-means++', 'ckm++']
                        (default: 'rnd')
        num_init: Number of different initial starting configurations,
                    i.e. different sets of initial centers (default: 8).
        max_iter: Maximum number of iterations (default: 100).
        distance: batched distance evaluator (default: LpDistance).
        p_norm: norm for lp distance (default: 2).
        tol: Relative tolerance with regards to Frobenius norm of the difference
                    in the cluster centers of two consecutive iterations to
                    declare convergence. (default: 1e-4)
        n_clusters: Default number of clusters to use if not provided in call
                (optional, default: 8).
        verbose: Verbosity flag to print additional info (default: True).
        seed: Seed to fix random state for randomized center inits
                (default: 123).
        n_priority_trials_before_fall_back: Number of trials trying to assign
                                            samples to constrained clusters based
                                            on priority values before falling back
                                            to assigning the node with the highest
                                            weight to a cluster which can still
                                            accommodate it or the dummy cluster
                                            otherwise. (default: 5)
        raise_infeasible: if set to False, will only display a warning
                            instead of raising an error (default: True)
        **kwargs: additional key word arguments for the distance function.
    """

    def __init__(
        self,
        model: VRPModel,
        depot_priority: bool = True,
        depot_priority_add: bool = True,
        # === CKMeans defaults ===
        normalize: Optional[str] = None,
        priority_scaling: str = "priority",
        nbh_knn: int = 16,
        convergence_criterion: str = "inertia",
        permute_k: bool = True,
        opt_last_frac: float = 0.0,
        opt_last_samples: int = 1,
        opt_last_prio: bool = True,
        # === ConstrainedKMeans defaults ===
        init_method: str = "rnd",
        num_init: int = 8,
        max_iter: int = 100,
        distance: BaseDistance = LpDistance,
        p_norm: int = 2,
        tol: float = 1e-4,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        n_priority_trials_before_fall_back: int = 5,
        raise_infeasible: bool = True,
        **kwargs
    ):
        super(RouteCKMeans, self).__init__(
            model=model,    # type: ignore
            normalize=normalize,
            priority_scaling=priority_scaling,
            vanilla_priority=False,
            pre_iter=0,     # does not work for VRP
            nbh_knn=nbh_knn,
            convergence_criterion=convergence_criterion,
            permute_k=permute_k,
            opt_last_frac=opt_last_frac,
            opt_last_samples=opt_last_samples,
            opt_last_prio=opt_last_prio,
            init_method=init_method,
            num_init=num_init,
            max_iter=max_iter,
            distance=distance,
            p_norm=p_norm,
            tol=tol,
            n_clusters=n_clusters,
            verbose=verbose,
            seed=seed,
            n_priority_trials_before_fall_back=n_priority_trials_before_fall_back,
            raise_infeasible=raise_infeasible,
            **kwargs,
        )
        self.depot_priority = depot_priority
        self.depot_priority_add = depot_priority_add

    def _check_weights(
        self,
        weights,
        dims: Tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Override since depot weights should be 0, which is not allowed normally."""
        ###
        if not isinstance(weights, Tensor):
            raise TypeError(
                f"weights has to be a torch.Tensor " f"but got {type(weights)}."
            )
        if not ((0 <= weights) & (weights <= 1)).all():
            raise ValueError(
                "weights must be positive and " "be normalized between [0, 1]"
            )
        bs, n, d = dims
        if len(weights.shape) == 2:
            if weights.size(0) != bs or weights.size(1) != n:
                raise ValueError(
                    f"weights needs to be of shape "
                    f"({bs}, {n}, ),"
                    f"but got {tuple(weights.shape)}."
                )
        else:
            raise ValueError(
                f"weights have unsupported shape of "
                f"{tuple(weights.shape)} "
                f"instead of ({bs}, {n})."
            )
        return weights.contiguous().to(dtype=dtype, device=device)

    @torch.no_grad()
    def _cluster(
            self, x: Tensor, centers: Tensor, k: LongTensor, weights: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Execute main algorithm.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )
            weights: normalized weights w.r.t. constraint of 1.0 (BS, N, )
        """
        if self.seed is not None and self._gen is None:
            self._gen = torch.Generator(device=x.device)
            self._gen.manual_seed(self.seed)

        weights = self._check_weights(
            weights, dims=x.shape, dtype=x.dtype, device=x.device
        )
        bs, n, d = x.size()
        # add dummy center at origin to assign nodes which cannot be assigned
        # at an intermediate point because they violate all capacities
        centers = torch.cat(
            (centers, torch.zeros((bs, self.num_init, 1, d), device=x.device)), dim=2
        )
        k_mask, k_max_range = self._get_kmask(k, num_init=self.num_init)
        k_max = centers.size(2)
        self.score_fn = self.score_fn.to(device=x.device)

        bsm = bs * self.num_init
        weights = weights[:, None, :].expand(bs, self.num_init, n).reshape(bsm, n)

        # create graph
        nodes = torch.cat((
            x[:, None, :, :].expand(bs, self.num_init, n, -1).reshape(bsm, n, -1),
            weights[:, :, None]
        ), dim=-1)
        edges, edges_w = knn_graph(x, knn=self.nbh_knn, device=x.device,
                                   num_init=self.num_init, vrp=True)
        node_emb = None

        # split depot coords and remove weight
        # the depot features are still retained in the nodes and edges features
        depot_x = x[:, 0]
        x_no_depot = x[:, 1:]  # w/o depot!
        weights = weights[:, 1:]  # w/o depot!
        depot_dist = None
        if self.depot_priority:
            # compute distance to depot
            # (bs, n, d), (bs, num_init, k_max, d)
            depot_dist = self._pairwise_distance(
                x_no_depot, depot_x[:, None, None, :].expand(-1, self.num_init, 1, -1)
            ).view(bsm, n-1)

        i, change = 0, None
        inertia = torch.empty((bs, self.num_init), device=x.device).fill_(float("inf"))
        c_assign_best, centers_best, inertia_best = None, None, None
        n_iters = self.max_iter-self.pre_iter
        for i in range(n_iters):
            centers[k_mask] = float("inf")
            old_centers = centers.clone()
            old_inertia = inertia
            # get cluster assignments
            c_assign, node_emb, inertia, centers = self._assign(
                x_no_depot, centers, weights, k_mask,
                nodes=nodes,
                edges=edges,
                edges_w=edges_w,
                node_emb=node_emb,
                depot_x=depot_x,
                depot_dist=depot_dist,
            )
            if centers is None:
                # update cluster centers
                # add depot
                depot_assign = k_max_range[:, None, :].expand(bs, c_assign.size(1), -1)
                centers = group_by_label_mean(
                    x=torch.cat((
                        depot_x[:, None, :].expand(-1, k_max, -1),
                        x_no_depot
                    ), dim=1),
                    labels=torch.cat((
                        depot_assign,
                        c_assign
                    ), dim=-1),
                    k_max_range=k_max_range
                )
            if inertia is None:
                # compute inertia
                inertia = self._inertia_with_inf(x_no_depot, c_assign, centers)
                if self.depot_priority:
                    # (bs, n, d), (bs, num_init, k_max, d)
                    center_depot_dist = self._pairwise_distance(
                        depot_x[:, None, :], centers
                    ).reshape(bs, self.num_init, -1)
                    # set dist to dummy centers to 0
                    center_depot_dist[k_mask] = 0
                    # long shallow clusters are better for tours, so the farther away the center
                    # from the depot, the better
                    inertia -= center_depot_dist.pow(2).sum(-1)
            # update best found so far
            if c_assign_best is None:
                c_assign_best = c_assign
                centers_best = centers
                inertia_best = inertia
            else:
                better = inertia < inertia_best
                c_assign_best = torch.where(better[:, :, None], c_assign, c_assign_best)
                centers_best = torch.where(better[:, :, None, None], centers, centers_best)
                inertia_best = torch.where(better, inertia, inertia_best)

            # check convergence criterion
            if self.tol is not None:
                if self.convergence_criterion == "shift":
                    change = self._calculate_shift(centers, old_centers, p=self.p_norm)
                elif self.convergence_criterion == "inertia":
                    if (old_inertia < float("inf")).all():
                        change = torch.abs(inertia-old_inertia)
                    else:
                        change = old_inertia
                else:
                    raise RuntimeError()
                if (change < self.tol).all():
                    if self.verbose:
                        print(
                            f"Full batch converged at iteration "
                            f"{i + 1}/{self.max_iter} with change: "
                            f"{change.view(-1, self.num_init).mean(-1).cpu().item()}."
                        )
                    break

        if self.tol is not None and self.verbose and i == n_iters - 1:
            print(
                f"No convergence after "
                f"{self.max_iter} maximum iterations."
                f"\nThere were some changes in last iteration "
                f"larger than specified threshold {self.tol}: "
                f"\n{change.max()}"
            )

        # do final assignment step
        centers[k_mask] = float("inf")
        c_assign, _, inertia, _ = self._assign(
            x_no_depot, centers, weights, k_mask,
            nodes=nodes,
            edges=edges,
            edges_w=edges_w,
            node_emb=node_emb,
            depot_x=depot_x,
            depot_dist=depot_dist,
        )
        if (c_assign < 0).any():
            # There remain some dummy clusters after convergence.
            # This means the algorithm could not find a
            # feasible assignment for at least one init
            # Check if there is at least 1 feasible solution for each instance
            feasible = (c_assign >= 0).all(-1).any(-1)
            if not feasible.all():
                inf_idx = (feasible == 0).nonzero().squeeze()
                msg = (
                    f"No feasible assignment found for "
                    f"instance(s) at idx: {inf_idx}.\n"
                    f"(Try to increase the number of clusters "
                    f"or loosen the constraints.)"
                )
                if self.raise_infeasible:
                    raise InfeasibilityError(msg)
                else:
                    warn(msg + "\nInfeasible instances removed from output.")

        if inertia is None:
            # compute inertia
            inertia = self._inertia_with_inf(x_no_depot, c_assign, centers)
            if self.depot_priority:
                # (bs, n, d), (bs, num_init, k_max, d)
                center_depot_dist = self._pairwise_distance(
                    depot_x[:, None, :], centers
                ).reshape(bs, self.num_init, -1)
                # set dist to dummy centers to 0
                center_depot_dist[k_mask] = 0
                # long shallow clusters are better for tours, so the farther away the center
                # from the depot, the better
                inertia -= center_depot_dist.pow(2).sum(-1)
        better = inertia < inertia_best
        if self.verbose:
            print(f"last assignment was best for: {better.cpu().numpy()}")
        c_assign_best = torch.where(better[:, :, None], c_assign, c_assign_best)
        centers_best = torch.where(better[:, :, None, None], centers, centers_best)
        inertia_best = torch.where(better, inertia, inertia_best)
        # select best init
        best_init = torch.argmin(inertia_best, dim=-1)
        b_idx = torch.arange(bs, device=x.device)
        return (
            c_assign_best[b_idx, best_init],
            centers_best[b_idx, best_init],
            inertia_best[b_idx, best_init],
            None,
        )

    def _assign(
            self,
            x: Tensor,
            centers: Tensor,
            weights: Tensor,
            k_mask: Tensor,
            nodes: Optional[Tensor] = None,
            edges: Optional[Tensor] = None,
            edges_w: Optional[Tensor] = None,
            node_emb: Optional[Tensor] = None,
            depot_x: Optional[Tensor] = None,
            depot_dist: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[LongTensor, Tensor, Any, Any]:
        """
        Assignment step using score function
        instead of standard priority rule.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            weights: normalized weights w.r.t. constraint of 1.0 (BSM, N, )
            k_mask: (BS, num_init, k_max)
            nodes: (BSM, N, D+1)
            edges: (2, M)
            edge_w: (M, )
            node_emb:
            depot_dist: (BS, N, 1)

        """
        bs, num_init, k_max, _ = centers.size()
        n = x.size(1)
        bsm = bs * num_init
        b_idx = torch.arange(bsm, device=x.device)
        # dist: (bs, num_init, n, k_max) -> (bs*num_init, k_max, n)
        dist = self._pairwise_distance(x, centers).view(bsm, n, k_max).permute(0, 2, 1)

        medoids = torch.argmin(dist, dim=-1)
        priority, node_emb = self.score_fn(
            nodes=nodes,
            centroids=centers.view(bsm, k_max, -1),
            medoids=medoids,
            c_mask=k_mask.reshape(bsm, -1),
            edges=edges,
            weights=edges_w,
            node_emb=node_emb,
        )
        # make sure all values are positive
        priority += torch.abs(priority.min())
        if self.priority_scaling == "priority":
            # multiply priority scores by heuristic weights
            if self.depot_priority:
                assert depot_dist is not None
                if self.depot_priority_add:
                    priority *= (
                        (weights[:, None, :].expand(bsm, k_max, n) / dist) +
                        depot_dist[:, None, :].expand(-1, k_max, -1)
                    )
                else:
                    priority *= (
                            (weights[:, None, :].expand(bsm, k_max, n) / dist) *
                            depot_dist[:, None, :].expand(-1, k_max, -1)
                        )
            else:
                priority *= (weights[:, None, :].expand(bsm, k_max, n) / dist)
        else:
            raise NotImplementedError(self.priority_scaling)

        # mask dummy centers
        priority[
            k_mask[:, :, :, None].expand(bs, num_init, k_max, n).reshape(bsm, k_max, n)
        ] = -float("inf")

        inertia, centers = None, None
        assignment = -torch.ones((bsm, n), device=x.device, dtype=torch.long)
        cluster_capacity = torch.ones(k_mask.size(), device=x.device)
        cluster_capacity[k_mask] = 0
        cluster_capacity = cluster_capacity.view(bsm, k_max)
        nc = (~k_mask).reshape(bsm, -1).sum(-1)
        ci = iter(CycleIdx(idx_lens=nc, gen=self._gen, permute=self.permute_k))
        not_assigned = (assignment < 0)

        i = 0
        # last opt_last_frac % of unassigned nodes
        last_pp = n - np.ceil(n * self.opt_last_frac) if self.opt_last_frac > 0 else 0
        while not_assigned.any():
            # get idx of center
            c_idx = next(ci)
            # select for current center idx
            remaining_cap = cluster_capacity[b_idx, c_idx].clone()
            prios = priority[b_idx, c_idx]
            # set priority to -inf for nodes which were already assigned
            # or which cannot be accommodated in terms of weights
            msk = (~not_assigned) | (weights > remaining_cap[:, None])
            prios[msk] = -float("inf")

            # check if valid assignment is possible
            valid_idx = ~((prios == -float("inf")).all(-1))
            if valid_idx.any():
                n_idx = prios.max(dim=-1).indices[valid_idx]
                c_idx = c_idx[valid_idx]
                # do assignment
                assignment[valid_idx, n_idx] = c_idx
                # adjust cluster capacity
                cluster_capacity[valid_idx, c_idx] = remaining_cap[valid_idx] - weights[valid_idx, n_idx]
                not_assigned = (assignment < 0)

            i += 1
            if 1 < last_pp < i:
                # the idea is, that the last few assignments are the most difficult,
                # since they have to cope with highly constrained center capacities
                # such that to rely on the pre defined order of the cyclic index
                # (which until this point has ensured that approx. the same number
                # of nodes was assigned to each cluster)
                # can lead to sub-optimal assignments in case some clusters have many
                # nodes with very large or very small weights, so we use a different strategy here
                n_na = not_assigned.sum(-1)
                n_na_max = n_na.max()
                dist = dist.permute(0, 2, 1)
                # pad weights and indices of unassigned nodes
                na_msk = (torch.arange(n_na_max, device=x.device)[None, :].expand(bsm, -1) < n_na[:, None])
                na_w = torch.zeros((bsm, n_na_max), device=x.device)
                na_idx = -torch.ones((bsm, n_na_max), device=x.device, dtype=torch.long)
                na_w[na_msk] = weights[not_assigned]
                na_idx[na_msk] = not_assigned.nonzero()[:, 1]
                if self.opt_last_prio:
                    na_prio = torch.zeros((bsm, n_na_max), device=x.device)
                    na_prio[na_msk] = priority.max(dim=1).values[not_assigned]
                if self.opt_last_samples > 1:
                    # sample different assignments for last unassigned nodes
                    # based on their weight and select the best one
                    bsms = bsm * self.opt_last_samples
                    if self.opt_last_prio:
                        probs = na_prio
                    else:
                        probs = na_w.clone()
                    if self.depot_priority:
                        # add the depot dist to the weights
                        probs[na_msk] += depot_dist[not_assigned]
                    probs[probs <= 0] = -float("inf")
                    probs = torch.softmax(probs, dim=-1)
                    na_w = na_w[:, None, :].expand(-1, self.opt_last_samples, -1).reshape(bsms, n_na_max)
                    na_idx = na_idx[:, None, :].expand(-1, self.opt_last_samples, -1).reshape(bsms, n_na_max)
                    probs = probs[:, None, :].expand(-1, self.opt_last_samples, -1)
                    # sample
                    idx_select = torch.multinomial(
                        probs.reshape(bsms, n_na_max),
                        num_samples=n_na_max,
                        replacement=False,
                        generator=self._gen,
                    )
                    # select from original node indices
                    na_idx = na_idx.gather(index=idx_select, dim=-1)
                    na_w = na_w.gather(index=idx_select, dim=-1)
                    # set cap of dummy cluster to 1
                    cluster_capacity[:, -1] = 1
                    cluster_capacity = cluster_capacity[:, None, :] \
                        .expand(-1, self.opt_last_samples, -1) \
                        .reshape(bsms, k_max).clone()
                    dist = dist[:, None, :, :] \
                        .expand(-1, self.opt_last_samples, -1, -1) \
                        .reshape(bsms, n, k_max)
                    assignment = assignment[:, None, :] \
                        .expand(-1, self.opt_last_samples, -1) \
                        .reshape(bsms, n).clone()
                    b_idx = torch.arange(bsms, device=x.device)
                else:
                    # do a greedy assignment based on weights:
                    # sort in order of decreasing weight
                    if self.opt_last_prio:
                        _, sort_idx = na_prio.sort(descending=True)
                        na_w = na_w.gather(index=sort_idx, dim=-1)
                    else:
                        # do a greedy assignment based on weights:
                        # sort in order of decreasing weight
                        na_w, sort_idx = na_w.sort(descending=True)
                    # apply to idx
                    na_idx = na_idx.gather(index=sort_idx, dim=-1)
                    # set cap of dummy cluster to 1
                    cluster_capacity[:, -1] = 1
                # select unassigned nodes according to weight
                # and assign to closest cluster which can still accommodate them
                for j in range(n_na_max):
                    idx = na_idx[:, j]
                    c_dist = dist[b_idx, idx]
                    c_dist, sort_idx = c_dist.sort()
                    w = na_w[:, j]
                    valid = cluster_capacity >= w[:, None]
                    # apply sorting
                    valid = valid.gather(index=sort_idx, dim=-1)
                    # select first valid idx
                    v_idx = torch.argmax(valid.float(), dim=-1)
                    # map back
                    v_idx = sort_idx[b_idx, v_idx]
                    idx_msk = (idx < 0)
                    if idx_msk.any():
                        v_idx[idx_msk] = -1  # dummy cluster idx
                        idx_msk = ~idx_msk  # only assign where still unassigned nodes remaining
                        assignment[b_idx[idx_msk], idx[idx_msk]] = v_idx[idx_msk]
                    else:
                        assignment[b_idx, idx] = v_idx
                    cluster_capacity[b_idx, v_idx] = cluster_capacity[b_idx, v_idx] - w
                    # reset cap of dummy cluster to 1
                    cluster_capacity[:, -1] = 1

                assert (assignment >= 0).all()
                if self.opt_last_samples > 1:
                    # select best samples
                    k_max_range = torch.arange(k_max, device=x.device)
                    assignment = assignment.reshape(bs, -1, n)
                    assignment[assignment >= k_max - 1] = -1
                    # add depot to each group for center calculation
                    depot_assign = k_max_range[None, None, :].expand(bs, assignment.size(1), -1)
                    centers = group_by_label_mean(
                        x=torch.cat((
                            depot_x[:, None, :].expand(-1, k_max, -1),
                            x
                        ), dim=1),
                        labels=torch.cat((
                            depot_assign,
                            assignment
                        ), dim=-1),
                        k_max_range=k_max_range[None, :].expand(bs, -1)
                    )
                    # compute inertia
                    inertia = self._inertia_with_inf(x, assignment, centers).view(bsm, self.opt_last_samples)
                    if self.depot_priority:
                        # (bs, n, d), (bs, num_init, k_max, d)
                        center_depot_dist = self._pairwise_distance(
                            depot_x[:, None, :], centers
                        ).reshape(bs, self.num_init, self.opt_last_samples, -1)
                        # set dist to dummy centers to 0
                        center_depot_dist[k_mask[:, :, None, :].expand(-1, -1, self.opt_last_samples, -1)] = 0
                        # long shallow clusters are better for tours, so the farther away the center
                        # from the depot, the better
                        inertia -= center_depot_dist.pow(2).sum(-1).view(bsm, self.opt_last_samples)

                    inertia, best_idx = inertia.min(dim=-1)
                    b_idx = b_idx[:bsm]
                    assignment = assignment.view(bsm, self.opt_last_samples, -1)[b_idx, best_idx]
                    centers = centers.view(bsm, self.opt_last_samples, k_max, -1)[b_idx, best_idx].view(bs, num_init,
                                                                                                        k_max, -1)
                    inertia = inertia.view(bs, -1)

                # stop loop
                break

            # from the point on, when all nodes should be assigned
            # assuming perfect allocation, we check if there are
            # instances which cannot be solved under the current assignment
            # and assign remaining nodes to the dummy center if that is the case
            if i > n + 2 * k_max + 1:
                msk = not_assigned.sum(-1) > 0
                if msk.any():
                    # check infeasibility
                    remaining_cap = cluster_capacity[msk].max(-1).values
                    for idx, w, na, rc in zip(msk.nonzero(), weights[msk], not_assigned[msk], remaining_cap):
                        if w[na].max() > rc:
                            assignment[idx, na] = k_max-1
                    not_assigned = (assignment < 0)

        # replace k_max idx with -1 for following computations
        assignment[assignment >= k_max - 1] = -1

        if self.verbose:
            print(f"assignment step finished after {i} sub-iterations.")

        return assignment.view(bs, num_init, n), node_emb, inertia, centers

    def inference(
            self,
            seed: int,
            instance: CCPInstance,
            k: Optional[int] = None,
            cuda: bool = False,
            num_cores: int = 1,
            k_search_bs: int = 8,
            k_search_iter: int = 16,
            k_search_init: int = 4,
            min_k: bool = True,
            **kwargs
    ):
        """Convenience function for test inference."""
        coords = instance.coords
        weights = instance.demands
        # cast to tensor
        coords = torch.from_numpy(coords) if isinstance(coords, np.ndarray) else coords
        weights = torch.from_numpy(weights) if isinstance(weights, np.ndarray) else weights
        # manage device
        device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        coords = coords.unsqueeze(0).to(dtype=torch.float, device=device)
        weights = weights.unsqueeze(0).to(dtype=torch.float, device=device)

        if self.score_fn is not None:
            self.score_fn = self.score_fn.to(device=device)
        self.seed = seed

        is_inf = False
        k_large = False
        org_opt_last_frac = self.opt_last_frac
        org_init = self.num_init
        assignment, solution = None, None
        t_start = default_timer()
        if k is not None:
            if isinstance(k, np.ndarray):
                k = torch.from_numpy(k)
                k_large = (k > 100).any()
            elif isinstance(k, torch.Tensor):
                k_large = (k > 100).any()
            else:
                k = int(k)
                k_large = k > 100

            n = instance.graph_size
            if k_large:
                self.opt_last_frac = max(0, min(1 - (k/n)*1.1, 0.9))
                self.num_init = min(4, self.num_init)
                k_search_init = 1

            try:
                assignment = self(x=coords, weights=weights, k=k).labels
            except InfeasibilityError as ie:
                is_inf = True
                warn(f"Error: {ie}")
                assignment = None
                k += 1
            self.num_init = org_init

        if k is None or is_inf:
            # use less iters to find k
            org_max_iter, org_init, org_samples = self.max_iter, self.num_init, self.opt_last_samples
            self.max_iter = min(k_search_iter, self.max_iter)
            self.num_init = min(k_search_init, self.num_init)
            self.opt_last_samples = 1   # search greedily
            self.raise_infeasible = False
            if k is None:
                # a good proxy for the minimal k to start with is the min number
                # of clusters necessary to serve all weights under perfect allocation
                k = int(np.ceil(np.sum(instance.demands)))
                k_large = k > 100

            if k_large:
                k_search_bs = max(2, k_search_bs // 2)
                self.num_init = max(1, min(k_search_init//2, self.num_init))
            trials = 0
            n = instance.graph_size
            max_trials = 2 * np.ceil(np.sqrt(n))

            x = coords.repeat((k_search_bs, 1, 1))
            w = weights.repeat((k_search_bs, 1))
            if k_large:
                inc = k_search_bs * 2
                try_k = torch.arange(k, k + inc, 2)
            else:
                inc = k_search_bs
                try_k = torch.arange(k, k + inc)

            while True:
                if self.verbose:
                    print(f"try: k={try_k} (inc: {inc})")
                try:
                    result = self(x=x, weights=w, k=try_k)
                    assignment = result.labels
                    inertia = result.inertia
                except InfeasibilityError:
                    print("x")
                    assignment = None
                    inertia = None
                if assignment is not None:
                    if (assignment >= 0).all(dim=-1).any():
                        break
                    else:
                        try_k += inc
                        trials += k_search_bs
                else:
                    try_k += inc
                    trials += k_search_bs
                if trials >= max_trials:
                    break

            if min_k:
                # get the smallest k with a feasible assignment
                k_idx = (assignment < 0).any(dim=-1).float().argmin()
            else:
                # simply use best
                k_idx = inertia.argmin()
            k = int(try_k[k_idx])
            if self.verbose:
                print(f"Found feasible k. Running final optimization with k={k}.")

            # reset to original value
            self.max_iter = org_max_iter
            self.num_init = min(4, self.num_init) if k_large else org_init
            self.opt_last_samples = org_samples
            # run algorithm completely with original params
            try:
                assignment = self(x=coords, weights=weights, k=k).labels
            except InfeasibilityError:
                assignment = None

        if assignment is not None:
            assert assignment.size(0) == 1
            assignment = assignment[0]
            solution, obj = self.route_assignment(
                assignment,
                instance.coords,
                num_cores=num_cores
            )
            if self.verbose:
                print(f"routed solution with objective: {obj}")

        t_total = default_timer() - t_start

        self.num_init = org_init
        self.opt_last_frac = org_opt_last_frac
        return solution, t_total

    @staticmethod
    def _solve_tsp(args: Tuple):
        assign, points, idx = args
        idx_selection = np.concatenate((
            np.zeros(1, dtype=int),  # depot
            (np.nonzero(assign == idx)[0] + 1)  # since depot idx = 0
        ), axis=-1)
        idx_coords = points[idx_selection]
        dist_mat = calc_distance_matrix(idx_coords, idx_coords, p=2)
        sol, obj = solve_tsp_fast(dist_mat, list(range(len(idx_coords))))
        # map back to original indices
        return (idx_selection[np.array(sol)].tolist(), obj)

    def route_assignment(self, assignment: Tensor, coords: np.ndarray, num_cores: Optional[int] = None):
        """Route nodes of assigned groups, i.e. solve corresponding TSP."""
        # setup parallel processing
        phys_cores = psutil.cpu_count(logical=False)
        if num_cores is None or num_cores <= 0:
            num_cores = phys_cores
        else:
            num_cores = min(num_cores, phys_cores)

        assignment = assignment.cpu().numpy()
        nc = assignment.max() + 1
        if num_cores == 1:
            solution = []
            total_dist = 0
            for i in range(nc):
                idx_selection = np.concatenate((
                    np.zeros(1, dtype=int),     # depot
                    (np.nonzero(assignment == i)[0] + 1)  # since depot idx = 0
                ), axis=-1)
                idx_coords = coords[idx_selection]
                dist_mat = calc_distance_matrix(idx_coords, idx_coords, p=2)
                sol, obj = solve_tsp_fast(dist_mat, list(range(len(idx_coords))))
                # map back to original indices
                solution.append(idx_selection[np.array(sol)].tolist())
                total_dist += obj
            return solution, total_dist
        else:
            with Pool(num_cores) as pool:
                result_tuples = list(
                    pool.imap(
                        self._solve_tsp,
                        [(assignment, coords, i) for i in range(nc)]
                    )
                )
            assert len(result_tuples) == nc
            return [tup[0] for tup in result_tuples], sum([tup[1] for tup in result_tuples])
