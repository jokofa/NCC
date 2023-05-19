#
from typing import Union, Any, Optional, Tuple
from warnings import warn
from timeit import default_timer
import numpy as np
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_kmeans.utils.distances import BaseDistance, LpDistance
from torch_kmeans.utils.utils import first_nonzero, group_by_label_mean
from torch_kmeans.clustering.constr_kmeans import ConstrainedKMeans, InfeasibilityError

from lib.utils import CCPInstance
from lib.ltr.ccp.ccp_model import CCPModel
from lib.ltr.utils import knn_graph, CycleIdx


class NCCNA(ConstrainedKMeans):
    """
    without alternative assignment

    Args:
        model: learned score function estimator
        normalize: String id of method to use to normalize input.
                    one of ['mean', 'minmax', 'unit'].
                    None to disable normalization. (default: None).
        priority_scaling: method to use for priority scaling
        vanilla_priority: only use heuristic priorities without score function
        pre_iter: number of prior standard constrained kmeans
                    iterations to initialize centers (default: 2)
        nbh_knn: size of KNN graph neighborhood to consider in model (default: 16)
        convergence_criterion: criterion to check for convergence
                                one of ['shift', 'inertia'], (default: 'inertia')
        num_samples: how many rollouts to sample
        relocate_iter:  NOT IMPLEMENTED
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
    INIT_METHODS = ["rnd", "k-means++", "topk", "ckm++", "c2km++"]
    NORM_METHODS = ["mean", "minmax", "unit"]
    CONVERGENCE = ["shift", "inertia"]
    SCALING = ["priority", "regret", "weights"]

    def __init__(
        self,
        model: CCPModel,
        normalize: Optional[str] = None,
        priority_scaling: str = "priority",
        vanilla_priority: bool = False,
        pre_iter: int = 0,
        nbh_knn: int = 16,
        convergence_criterion: str = "inertia",
        permute_k: bool = True,
        num_samples: int = 1,
        relocate_iter: int = 0,
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
        super(NCCNA, self).__init__(
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
        self.score_fn = None
        if not vanilla_priority:
            model.eval()    # make sure model is in eval mode
            self.score_fn = model
        self.normalize = normalize
        if self.normalize is not None and self.normalize not in self.NORM_METHODS:
            raise ValueError(
                f"unknown <normalize> method: {self.normalize}. "
                f"Please choose one of {self.NORM_METHODS}"
            )
        if priority_scaling is not None:
            assert priority_scaling.lower() in self.SCALING
            self.priority_scaling = priority_scaling.lower()
        else:
            self.priority_scaling = ""
        self.vanilla_priority = vanilla_priority
        assert pre_iter < self.max_iter
        self.pre_iter = pre_iter
        self.nbh_knn = nbh_knn
        self.convergence_criterion = convergence_criterion.lower()
        assert self.convergence_criterion in self.CONVERGENCE
        self.permute_k = permute_k
        self.num_samples = num_samples
        self.relocate_iter = min(relocate_iter, self.max_iter-1)
        self._gen = None

    def fit(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        raise NotImplementedError()

    def predict(self, x: Tensor, weights: Tensor, **kwargs) -> LongTensor:
        raise NotImplementedError()

    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> LongTensor:
        raise NotImplementedError()

    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs):
        if self.init_method == "rnd":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        elif self.init_method == "topk":
            return self._init_topk(x, k, **kwargs)
        elif self.init_method == "ckm++":
            return self._init_ckm_plus(x, k, **kwargs)
        elif self.init_method == "c2km++":
            assert "weights" in list(kwargs.keys())
            w = kwargs.pop("weights", None)
            return self._init_ckm_plus(x, k, weights=(w+1)**2, **kwargs)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

    @staticmethod
    @torch.jit.script
    def _inertia_d0(x: Tensor, centers: Tensor, labels: Tensor, k_max: int) -> Tensor:
        """Compute inertia for inputs of dim (BS, N, D),
        i.e. without considering any num_init or sample dimensions!"""
        bs, n, d = x.size()
        centers = centers.view(-1, k_max, d)
        labels = labels.view(-1, n)
        assert bs == centers.size(0) == labels.size(0)
        return (torch.norm((
            x -
            centers.gather(
                index=labels[:, :, None].expand(-1, -1, d),
                dim=1,
            )
        ), p=2, dim=-1) ** 2).sum(-1)

    def _inertia_with_inf(self, x: Tensor, c_assign: Tensor, centers: Tensor):
        """Compute inertia over full batch possibly including infeasible assignments."""
        infeasible = (c_assign < 0)
        if infeasible.any():
            # There are some dummy clusters in current iter.
            # This means the algorithm could not find a
            # feasible assignment for at least one init
            feasible = (~infeasible).all(-1)
            inertia = torch.empty(list(centers.shape)[:2],
                                  device=x.device).fill_(float("inf"))
            if feasible.any():
                inertia[feasible] = self._inertia_d0(
                    x=x[:, None, :, :].expand(-1, centers.size(1), -1, -1)[feasible],
                    centers=centers[feasible],
                    labels=c_assign[feasible],
                    k_max=centers.size(2)
                )
            return inertia
        else:
            return self._calculate_inertia(x, centers, c_assign)

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
        if self.score_fn is not None:
            self.score_fn = self.score_fn.to(device=x.device)

        bsm = bs * self.num_init
        weights = weights[:, None, :].expand(bs, self.num_init, n).reshape(bsm, n)

        # run some prior iterations of standard algorithm for better center positions
        for i in range(self.pre_iter):
            centers[k_mask] = float("inf")
            # get cluster assignments
            c_assign = self._pre_assign(x, centers, weights, k_mask)
            # update cluster centers
            centers = group_by_label_mean(x, c_assign, k_max_range)

        # create graph
        nodes = torch.cat((
            x[:, None, :, :].expand(bs, self.num_init, n, -1).reshape(bsm, n, -1),
            weights[:, :, None]
        ), dim=-1)
        edges, edges_w = knn_graph(x, knn=self.nbh_knn, device=x.device, num_init=self.num_init)
        node_emb = None

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
                x, centers, weights, k_mask,
                nodes=nodes,
                edges=edges,
                edges_w=edges_w,
                node_emb=node_emb,
            )
            if centers is None:
                # update cluster centers
                centers = group_by_label_mean(x, c_assign, k_max_range)
            if inertia is None:
                # compute inertia
                inertia = self._inertia_with_inf(x, c_assign, centers)
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
                            f"{change.view(-1, self.num_init).mean(-1).cpu()}."
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
            x, centers, weights, k_mask,
            nodes=nodes,
            edges=edges,
            edges_w=edges_w,
            node_emb=node_emb,
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
            inertia = self._inertia_with_inf(x, c_assign, centers)
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

        """
        bs, num_init, k_max, _ = centers.size()
        n = x.size(1)
        bsm = bs * num_init
        b_idx = torch.arange(bsm, device=x.device)
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers).view(bsm, n, k_max).permute(0, 2, 1)

        if self.vanilla_priority:
            priority = (weights[:, None, :].expand(bsm, k_max, n) / dist)
        else:
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
                # scale scores by weight/dist per center
                priority *= (weights[:, None, :].expand(bsm, k_max, n) / dist)
            elif self.priority_scaling == "regret":
                # scale scores by regret (Mulvey et al., 1984)
                # the absolute value difference between the
                # distance to the nearest and 2nd nearest center
                d = torch.topk(-dist, k=2, dim=1).values
                priority *= (torch.abs(d[:, 0, :]-d[:, 1, :]))[:, None, :]
            elif self.priority_scaling == "weights":
                priority *= weights[:, None, :].expand(bsm, k_max, n)

        # mask dummy centers
        priority[
            k_mask[:, :, :, None].expand(bs, num_init, k_max, n).reshape(bsm, k_max, n)
        ] = -float("inf")

        inertia, centers = None, None
        assignment = -torch.ones((bsm, n), device=x.device, dtype=torch.long)
        cluster_capacity = torch.ones(k_mask.size(), device=x.device)
        cluster_capacity[k_mask] = 0
        cluster_capacity = cluster_capacity.view(bsm, k_max)

        dist = dist.permute(0, 2, 1)
        #_idx = -torch.ones((bsm, n), device=x.device, dtype=torch.long)
        _idx = torch.arange(n, device=x.device, dtype=torch.long)[None, :].expand(bsm, n)
        w = weights
        # we treat the maximum over all cluster assignments
        # per node as the absolute priority for that node
        abs_prio = priority.max(dim=1).values
        if self.num_samples > 1:
            # sample different assignments for last unassigned nodes
            # based on their weight and select the best one
            bsms = bsm * self.num_samples
            probs = abs_prio
            probs[probs <= 0] = -float("inf")
            probs = torch.softmax(probs, dim=-1)
            w = w[:, None, :].expand(-1, self.num_samples, -1).reshape(bsms, -1)
            _idx = _idx[:, None, :].expand(-1, self.num_samples, -1).reshape(bsms, -1)
            probs = probs[:, None, :].expand(-1, self.num_samples, -1)
            # sample
            idx_select = torch.multinomial(
                probs.reshape(bsms, n),
                num_samples=n,
                replacement=False,
                generator=self._gen,
            )
            # select from original node indices
            _idx = _idx.gather(index=idx_select, dim=-1)
            w = w.gather(index=idx_select, dim=-1)
            # set cap of dummy cluster to 1
            cluster_capacity[:, -1] = 1
            cluster_capacity = cluster_capacity[:, None, :]\
                .expand(-1, self.num_samples, -1)\
                .reshape(bsms, k_max).clone()
            dist = dist[:, None, :, :]\
                .expand(-1, self.num_samples, -1, -1)\
                .reshape(bsms, n, k_max)
            assignment = assignment[:, None, :]\
                .expand(-1, self.num_samples, -1)\
                .reshape(bsms, n).clone()
            b_idx = torch.arange(bsms, device=x.device)
        else:
            # do a greedy assignment based on priority:
            _, sort_idx = abs_prio.sort(descending=True)
            w = w.gather(index=sort_idx, dim=-1)
            # apply to idx
            _idx = _idx.gather(index=sort_idx, dim=-1)
            # set cap of dummy cluster to 1
            cluster_capacity[:, -1] = 1
        # select unassigned nodes according to priority
        # and assign to closest cluster which can still accommodate them
        for j in range(n):
            idx = _idx[:, j]
            c_dist = dist[b_idx, idx]
            c_dist, sort_idx = c_dist.sort()
            wgt = w[:, j]
            valid = cluster_capacity >= wgt[:, None]
            # apply sorting
            valid = valid.gather(index=sort_idx, dim=-1)
            # select first valid idx
            v_idx = torch.argmax(valid.float(), dim=-1)
            # map back
            v_idx = sort_idx[b_idx, v_idx]
            idx_msk = (idx < 0)
            if idx_msk.any():
                v_idx[idx_msk] = -1     # dummy cluster idx
                idx_msk = ~idx_msk  # only assign where still unassigned nodes remaining
                assignment[b_idx[idx_msk], idx[idx_msk]] = v_idx[idx_msk]
            else:
                assignment[b_idx, idx] = v_idx
            cluster_capacity[b_idx, v_idx] = cluster_capacity[b_idx, v_idx] - wgt
            # reset cap of dummy cluster to 1
            cluster_capacity[:, -1] = 1

        not_assigned = (assignment < 0)
        if not_assigned.any():
            #raise RuntimeError()
            # all nodes should be assigned assuming perfect allocation,
            # we check here if there are instances which cannot be solved under the current assignment
            # and assign remaining nodes to the dummy center if that is the case
            msk = not_assigned.sum(-1) > 0
            if msk.any():
                if msk.any():
                    # check infeasibility
                    remaining_cap = cluster_capacity[msk].max(-1).values
                    for idx, w, na, rc in zip(msk.nonzero(), weights[msk], not_assigned[msk], remaining_cap):
                        if w[na].max() > rc:
                            assignment[idx, na] = k_max - 1
                    not_assigned = (assignment < 0)

        if self.num_samples > 1:
            # select best samples
            k_max_range = torch.arange(k_max, device=x.device)
            assignment = assignment.reshape(bs, -1, n)
            assignment[assignment == k_max - 1] = -1
            centers = group_by_label_mean(
                x,
                assignment,
                k_max_range[None, :].expand(bs, -1)
            )
            # compute inertia
            inertia = self._inertia_with_inf(x, assignment, centers).view(bsm, self.num_samples)
            inertia, best_idx = inertia.min(dim=-1)
            b_idx = b_idx[:bsm]
            assignment = assignment.view(bsm, self.num_samples, -1)[b_idx, best_idx]
            centers = centers.view(bsm, self.num_samples, k_max, -1)[b_idx, best_idx].view(bs, num_init, k_max, -1)
            inertia = inertia.view(bs, -1)

        # replace k_max idx with -1 for following computations
        assignment[assignment >= k_max-1] = -1

        if self.verbose:
            print(f"assignment step finished.")

        return assignment.view(bs, num_init, n), node_emb, inertia, centers

    def _pre_assign(
            self, x: Tensor, centers: Tensor, weights: Tensor, k_mask: Tensor, **kwargs
    ) -> LongTensor:
        """Execute standard constrained k-means assignment step."""
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers)
        bs, num_init, n, k_max = dist.size()
        bsm = bs * num_init
        dist = dist.view(bsm, n, k_max)

        # we use a heuristic approach to include the
        # cluster capacity by defining a priority value w.r.t. the weight
        # (demand, workload, etc.) of each point
        # The idea is to first assign points with a
        # relatively larger weight to the clusters
        # and then points with smaller weight which
        # can be more easily assigned to other clusters.
        priority = weights[:, :, None].expand(bsm, n, k_max) / dist
        priority[
            k_mask[:, :, None, :].expand(bs, num_init, n, k_max).reshape(bsm, n, k_max)
        ] = 0
        # loop over all nodes to sequentially assign them to clusters
        # while keeping track of cluster capacity
        assignment = -torch.ones((bsm, n), device=x.device, dtype=torch.long)
        cluster_capacity = torch.ones(k_mask.size(), device=x.device)
        cluster_capacity[k_mask] = 0
        cluster_capacity = cluster_capacity.view(bsm, k_max)
        for i in range(n):
            ##n_trials = min(n-i, self.n_trials)  # noqa
            n_trials = self.n_trials
            max_val_k, max_idx_k = priority.max(dim=-1)

            # select n_trials top priority nodes for each instance
            max_idx_n = max_val_k.topk(dim=-1, k=n_trials).indices
            # get corresponding cluster idx and weight
            cl_try = max_idx_k.gather(index=max_idx_n, dim=-1)
            w_try = weights.gather(index=max_idx_n, dim=-1)
            can_be_assigned = cluster_capacity.gather(index=cl_try, dim=-1) >= w_try
            # get first nonzero as idx and a validity mask
            # if any trial could be assigned
            valid_idx, fnz = first_nonzero(can_be_assigned, dim=-1)
            trial_select = fnz[valid_idx]
            cl_select = cl_try[valid_idx, trial_select]
            # do assignment
            n_select = max_idx_n[valid_idx, trial_select]
            assignment[valid_idx, n_select] = cl_select
            # mask priority of assigned nodes
            priority[valid_idx, n_select] = 0
            # adjust cluster capacity
            cur_cap = cluster_capacity[valid_idx, cl_select].clone()
            cluster_capacity[valid_idx, cl_select] = (
                    cur_cap - w_try[valid_idx, trial_select]
            )
            # all instances with no valid idx could not assign any trial node
            not_assigned = ~valid_idx

            if not_assigned.any():
                # complete current assignment where for some instances
                # all trials based on priority were not feasible,
                # by assigning the node with the highest weight
                # to a cluster which can still accommodate it
                # or the dummy cluster at the origin otherwise (idx = -1)
                n_not_assigned = not_assigned.sum()
                cur_cap = cluster_capacity[not_assigned].clone()
                available = assignment[not_assigned] < 0
                # select node with highest weight from remaining unassigned nodes
                try:
                    w = weights[not_assigned][available].view(n_not_assigned, -1)
                except RuntimeError:
                    # fallback: just select best of first min available clusters
                    sm = available.sum(-1)
                    min_av = sm.min()
                    av_msk = sm > min_av
                    if min_av <= 1:
                        av_valid, av_idx = first_nonzero(available)
                        available[av_msk] = False
                        available[av_msk, av_idx[av_msk]] = True
                    else:
                        avbl = available[av_msk]
                        bi_cp = 0
                        cnter = 0
                        for bi, zi in zip(*avbl.nonzero(as_tuple=True)):
                            if bi == bi_cp:
                                cnter += 1
                            else:
                                bi_cp += 1
                                cnter = 1
                            if cnter > min_av:
                                avbl[bi, zi] = False
                        available[av_msk] = avbl
                    w = weights[not_assigned][available].view(n_not_assigned, -1)

                max_w, max_idx = w.max(dim=-1, keepdims=True)
                max_idx_n = (
                    available.nonzero(as_tuple=True)[1]
                        .view(n_not_assigned, -1)
                        .gather(dim=-1, index=max_idx)
                        .squeeze(-1)
                )
                # check the cluster priorities of this node
                msk = cur_cap >= max_w
                n_prio_idx = (
                    priority[not_assigned, max_idx_n]
                        .sort(dim=-1, descending=True)
                        .indices
                )
                # reorder msk according to priority and select first valid index
                select_msk = msk.gather(index=n_prio_idx, dim=-1)
                # get first nonzero as idx
                valid_idx, fnz = first_nonzero(select_msk, dim=-1)
                # nodes which cannot be assigned to any cluster anymore
                # since no sufficient capacity is available
                # are assigned to a dummy cluster with idx -1.
                cl_select = -torch.ones(
                    n_not_assigned, device=x.device, dtype=torch.long
                )
                cl_select[valid_idx] = n_prio_idx[valid_idx, fnz[valid_idx]]
                # do assignment
                assignment[not_assigned, max_idx_n] = cl_select
                # adapt priority
                priority[not_assigned, max_idx_n] = 0
                # adjust cluster capacity
                cur_cap = cluster_capacity[not_assigned, cl_select].clone()
                cluster_capacity[not_assigned, cl_select] = cur_cap - max_w.squeeze(-1)

        return assignment.view(bs, num_init, n)

    def inference(
            self,
            seed: int,
            instance: CCPInstance,
            k: Optional[int] = None,
            cuda: bool = False,
            k_search_bs: int = 8,
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

        if k is not None:
            if isinstance(k, np.ndarray):
                k = torch.from_numpy(k)
            elif isinstance(k, torch.Tensor):
                k = k
            else:
                k = int(k)
            t_start = default_timer()
            try:
                assignment = self(x=coords, weights=weights, k=k).labels
            except InfeasibilityError:
                assignment = None
        else:
            org_max_iter = self.max_iter
            self.max_iter = 10
            self.raise_infeasible = False
            # a good proxy for the minimal k to start with is the min number
            # of clusters necessary to serve all weights under perfect allocation
            k = int(np.ceil(np.sum(instance.demands)))
            trials = 0
            n = instance.graph_size
            max_trials = 2 * np.ceil(np.sqrt(n))

            x = coords.repeat((k_search_bs, 1, 1))
            w = weights.repeat((k_search_bs, 1))
            try_k = torch.arange(k, k + k_search_bs)

            t_start = default_timer()
            while True:
                try:
                    result = self(x=x, weights=w, k=try_k)
                    assignment = result.labels
                    inertia = result.inertia
                except InfeasibilityError:
                    assignment = None
                    inertia = None
                if assignment is not None:
                    if (assignment >= 0).all(dim=-1).any():
                        break
                    else:
                        try_k += k_search_bs
                        trials += k_search_bs
                else:
                    try_k += k_search_bs
                    trials += k_search_bs
                if trials >= max_trials:
                    break

            if min_k:
                # get the smallest k with a feasible assignment
                k_idx = (assignment < 0).any(dim=-1).float().argmin()
            else:
                # use elbow method to decide about k
                v_msk = inertia < float("inf")
                if v_msk.sum() <= 2:
                    # simply use best
                    k_idx = inertia.argmin()
                else:
                    valid_inertia = inertia[v_msk]
                    diff = valid_inertia[:-1] - valid_inertia[1:]
                    avg = diff.median()
                    # the idx prior to the first idx at which
                    # the improvement is less than the median improvement
                    _k = max(0, (diff < avg).float().argmax()-1)
                    k_idx = torch.sum(~v_msk) + _k  # recover correct batch idx
            k = int(try_k[k_idx])
            self.max_iter = org_max_iter
            # run algorithm completely with original max iterations
            try:
                assignment = self(x=coords, weights=weights, k=k).labels
            except InfeasibilityError:
                assignment = None

        t_total = default_timer() - t_start
        # sanity feasibility check
        for i in torch.unique(assignment):
            if weights[assignment == i].sum() > 1:
                warn(f"there were infeasible assignments in final result!")

        assert assignment.size(0) == 1
        assignment = assignment[0]
        return assignment.cpu().numpy() if assignment is not None else assignment, t_total
