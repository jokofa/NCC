#
import os
import logging
import itertools as it
from warnings import warn
from copy import deepcopy
from typing import Optional, Tuple, Union, List, NamedTuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_kmeans.utils.utils import group_by_label_mean

from lib.utils.formats import CCPInstance, format_repr
from lib.problems import Generator, GraphNeighborhoodSampler
from lib.ltr.ccp.ccp_model import CCPModel
from lib.ltr.vrp.vrp_model import VRPModel

logger = logging.getLogger(__name__)


class CycleIdx:
    """Cyclic generator for different idx ranges."""
    def __init__(self,
                 idx_lens: torch.Tensor,
                 seed: Optional[int] = None,
                 gen: Optional[torch.Generator] = None,
                 permute: bool = True
                 ):
        assert len(idx_lens.shape) == 1
        self.bs = len(idx_lens)
        self.device = idx_lens.device
        self.seed = seed
        if gen is not None:
            self.gen = gen
        elif seed is not None:
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(self.seed)
        else:
            self.gen = None
        self.loops = None
        if (idx_lens == idx_lens[0]).all():
            self._all_same = True
            self._lens = idx_lens[0].cpu().item()
        else:
            self._all_same = False
            self._lens = idx_lens.cpu().numpy()

        if permute:
            self.permute()
        else:
            if self._all_same:
                self.loops = it.cycle(
                    torch.arange(
                        self._lens, device=self.device
                    )[:, None].expand(-1, self.bs)
                )
            else:
                self.loops = [it.cycle(torch.arange(l)) for l in idx_lens]

    def __iter__(self) -> Tensor:
        while True:
            if self._all_same:
                yield from self.loops
            else:
                yield torch.cat(
                    [next(i).unsqueeze(0) for i in self.loops],
                    dim=0
                )

    def permute(self):
        if self._all_same:
            self.loops = it.cycle(torch.cat([
                torch.randperm(self._lens, generator=self.gen, dtype=torch.long).unsqueeze(0)
                for _ in range(self.bs)
            ], dim=0).T)
        else:
            self.loops = [it.cycle(torch.randperm(l, generator=self.gen, dtype=torch.long)) for l in self._lens]


class InputTuple(NamedTuple):
    """Typed constrained clustering problem instance wrapper."""
    nodes: Tensor
    centroids: Tensor
    medoids: Tensor
    c_mask: Optional[Tensor] = None
    edges: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    node_emb: Optional[Tensor] = None
    assignment: Optional[Tensor] = None
    n_mask: Optional[Tensor] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)


@torch.jit.script
def cumsum0(t: torch.Tensor) -> torch.Tensor:
    """calculate cumsum of t starting at 0."""
    return torch.cat((
        torch.zeros(1, dtype=t.dtype, device=t.device),
        torch.cumsum(t, dim=-1)[:-1]
    ), dim=0)


def knn_graph(
        coords: Tensor,
        knn: int,
        device: torch.device = "cpu",
        num_init: int = 1,
        num_workers: int = 4,
        vrp: bool = False,
):
    bs, n, d = coords.size()
    # sample KNN edges for each node
    nbh_sampler = GraphNeighborhoodSampler(
        graph_size=n,
        k_frac=knn,
        num_workers=num_workers,
        vrp=vrp,
    )
    # starting node indices of each batch at first node
    bsm = bs * num_init
    ridx = cumsum0(
        torch.from_numpy(np.full(bsm, n))
            .to(dtype=torch.long, device=device)
    )
    stat_edges, stat_weights = [], []
    for i, c in enumerate(coords):
        e, w, _ = nbh_sampler(c)
        if num_init <= 1:
            stat_edges.append(e + ridx[i])  # increase node indices by running idx
            stat_weights.append(w)
        else:
            for j in range(num_init):
                stat_edges.append(e + ridx[(i*num_init)+j])
                stat_weights.append(w)

    edges = torch.stack(stat_edges, dim=1).view(2, -1)
    weights = torch.stack(stat_weights).view(-1)
    return edges, weights


def get_kmask(k: Tensor, num_init: int = 1) -> Tuple[Tensor, Tensor]:
    """Compute mask of number of clusters k for centers of each instance."""
    bs = k.size(0)
    # mask centers for which  k < k_max with inf to get correct assignment
    k_max = torch.max(k).cpu().item()
    k_max_range = torch.arange(k_max, device=k.device)[None, :].expand(bs, -1)
    k_mask = k_max_range >= k[:, None]
    k_mask = k_mask[:, None, :].expand(bs, num_init, -1)
    return k_mask, k_max_range


class NPZProblemDataset(Dataset):
    """Routing problem dataset wrapper."""
    def __init__(self,
                 npz_file_pth: str,
                 knn: int = 16,
                 ):
        """

        Args:
            npz_file_pth: path to numpy .npz dataset file
            knn: number of nearest neighbors for neighborhood graph
        """
        super(NPZProblemDataset, self).__init__()
        self.data_pth = npz_file_pth
        self.knn = knn

        self.size = None
        self.nbh_sampler = None
        self._file_path = None
        self._file_handle = None
        self._keys = None
        self._inst_cl = None
        self._load()

    def _load(self):
        f_ext = os.path.splitext(self.data_pth)[1]
        self._file_path = os.path.normpath(os.path.expanduser(self.data_pth))
        assert f_ext == ".npz"
        logger.info(f"Loading dataset from:  {self._file_path}")
        data = np.load(self._file_path)

        keys = data.files
        problem = data.get('problem', None)
        if isinstance(problem, np.ndarray):
            if problem.size > 0:
                if len(problem.shape) == 1:
                    problem = str(problem[0])
                else:
                    problem = str(problem)
            else:
                raise ValueError(f"problem: {problem}")
        problem = problem.lower()
        # cross checking with filepath
        if "vrp" in self._file_path.lower():
            problem_ = "vrp"
        else:
            problem_ = "ccp"
        if problem != problem_:
            warn(f"npz file specifies problem: '{problem}' but file path problem: '{problem_}'")
            problem = problem_
        ###
        size = data.get('size', None)
        if isinstance(size, np.ndarray):
            if size.size > 0:
                if len(size.shape) == 1:
                    size = int(size[0])
                else:
                    size = int(size)
            else:
                raise ValueError(f"size: {size}")
        self.size = size
        assert problem in Generator.COPS
        self._inst_cl = CCPInstance
        keys.remove('problem')
        keys.remove('size')
        self._keys = deepcopy(keys)
        self.data = {k: data[k] for k in self._keys}
        try:
            gs = data['graph_size'][0]
        except AttributeError:
            gs = 200
        self.nbh_sampler = GraphNeighborhoodSampler(
            graph_size=gs, k_frac=self.knn,
            num_workers=1, vrp=("vrp" in problem.lower()))

    def _prepare(self, x: CCPInstance):
        """prepare one instance"""
        e, w, _ = self.nbh_sampler(torch.from_numpy(x.coords))
        return (x, e, w)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._prepare(
            self._inst_cl(**{
                k: self.data[k][idx] for k in self._keys
            })
        )


def collate_batch(
        batch: List[Tuple[CCPInstance, Tensor, Tensor]],
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        knn: int = 16,
        vrp: bool = False,
) -> Tuple[InputTuple, Tensor]:
    """

    Args:
        batch: list of instances from dataloader
        device: computing device
        dtype: fp precision data type
        knn: number of nearest neighbors for neighborhood graph

    Returns:
        x: meta-instance with collated attributes
        y: corresponding assignment labels
    """
    assert isinstance(batch, List)
    bs = len(batch)
    b = batch[0]
    if isinstance(b, (Tuple, List)) and len(b) == 3:
        i, e, w = b
        # starting node indices of each batch at first node
        ridx = cumsum0(
            torch.from_numpy(np.full(bs, i.graph_size))
                .to(dtype=torch.long, device=device)
        )
    else:
        i = b
        e, w, ridx = None, None, None
    assert isinstance(i, CCPInstance)
    gs = i.graph_size
    cv = i.constraint_value
    coords = []
    demands = []
    labels = []
    num_components = []
    edges = []
    weights = []
    for i, b in enumerate(batch):
        if isinstance(b, (Tuple, List)) and len(b) == 3:
            inst, e, w = b
        else:
            inst = b
            e, w = None, None
        assert inst.graph_size == gs
        assert inst.constraint_value == cv
        assert len(inst.coords) == len(inst.demands) == gs
        assert inst.labels is not None
        coords.append(inst.coords)
        demands.append(inst.demands)
        nc = inst.num_components
        labels.append(inst.labels)
        if nc is None:
            nc = len(np.unique(inst.labels))
        num_components.append(nc)
        if e is not None and w is not None:
            edges.append(e + ridx[i])   # increase node indices by running idx
            weights.append(w)

    coords = torch.from_numpy(np.stack(coords)).to(device=device, dtype=dtype)
    demands = torch.from_numpy(np.stack(demands)).to(device=device, dtype=dtype)

    if len(edges) == len(weights) == bs:
        edges = torch.stack(edges, dim=1).view(2, -1).to(device=device)
        weights = torch.stack(weights).view(-1).to(device=device, dtype=dtype)
    else:
        edges, weights = knn_graph(coords, knn=knn, device=device, vrp=vrp)

    assert len(labels) == bs
    labels = torch.from_numpy(np.stack(labels)).to(device=device)
    assert len(num_components) == bs
    num_components = torch.tensor(num_components).to(device=device)

    k_max = num_components.max()
    k_msk, k_max_rng = get_kmask(num_components.clone(), 1)
    assert k_msk.size(1) == 1
    k_msk = k_msk.squeeze(1)
    # compute centers
    if vrp:
        gs = gs - 1
        # need to compute centers without depot
        centers = group_by_label_mean(
            coords[:, 1:, :],
            labels[:, None, :],
            k_max_rng.long()
        ).squeeze(1)
        # compute closest point, i.e. medoid but which cannot be depot!
        # medoid idx 0 will be considered first non-depot node
        medoids = torch.argmin(
            torch.norm(
                (coords[:, None, 1:, :].expand(-1, k_max, -1, -1) -
                 centers[:, :, None, :].expand(-1, -1, gs, -1)),
                p=2, dim=-1
            ), dim=-1
        )
    else:
        # compute centers (centroids)
        centers = group_by_label_mean(
            coords,
            labels[:, None, :],
            k_max_rng.long()
        ).squeeze(1)
        # compute closest point, i.e. medoid
        medoids = torch.argmin(
            torch.norm(
                (coords[:, None, :, :].expand(-1, k_max, -1, -1) -
                 centers[:, :, None, :].expand(-1, -1, gs, -1)),
                p=2, dim=-1
            ), dim=-1
        )

    nodes = torch.cat((coords, demands[:, :, None]), dim=-1)
    x = InputTuple(
        nodes=nodes,
        centroids=centers,
        medoids=medoids,
        c_mask=k_msk,
        edges=edges,
        weights=weights,
    )
    y = (
            labels[:, None, :].expand(bs, k_max, -1) ==
            torch.arange(k_max, device=device)[None, :, None].expand(bs, -1, gs)
    ).to(dtype=dtype)

    return x, y


def load_model(problem: str, ckpt_pth: str, **kwargs):
    """Loads and initializes the model from the specified checkpoint path."""
    print(f"loading model checkpoint: {ckpt_pth}")
    checkpoint = torch.load(ckpt_pth, **kwargs)
    cfg = checkpoint['hyper_parameters']['cfg']
    if problem.lower() == "ccp":
        mdl = CCPModel
    elif problem.lower() in ["vrp", "cvrp"]:
        mdl = VRPModel
    else:
        raise ValueError(f"unknown problem: '{problem}'")
    model = mdl(
        input_dim=cfg.input_dim,
        embedding_dim=cfg.embedding_dim,
        decoder_type=cfg.decoder_type,
        node_encoder_args=cfg.node_encoder_args,
        center_encoder_args=cfg.center_encoder_args,
        decoder_args=cfg.decoder_args,
    )
    sd = checkpoint['state_dict']
    # remove task model prefix if existing
    sd = {k[6:]: v for k, v in sd.items() if k[:6] == "model."}
    model.load_state_dict(sd)   # type: ignore
    return model
