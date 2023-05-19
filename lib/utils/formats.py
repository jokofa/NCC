#
from typing import NamedTuple, Union, List, Optional
import numpy as np
import torch

__all__ = ["RPInstance", "CCPInstance", "Obs", "SupObs"]


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"


class Obs(NamedTuple):
    """Named and typed tuple of observations."""
    bs: int
    n: int
    stat_node_features: torch.Tensor
    stat_edges: torch.LongTensor
    stat_weights: torch.Tensor

    assignment: torch.LongTensor
    pending: torch.BoolTensor
    nc: torch.LongTensor
    cluster_features: torch.Tensor
    cluster_mask: torch.BoolTensor
    node_i_idx: torch.LongTensor
    node_i_mask: torch.BoolTensor

    context_features: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))


class SupObs(NamedTuple):
    """Pseudo tuple of observations."""
    stat_node_features: torch.Tensor
    stat_edges: torch.LongTensor
    stat_weights: torch.Tensor

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))


class RPInstance(NamedTuple):
    """Typed routing problem instance wrapper."""
    coords: Union[np.ndarray, torch.Tensor]
    demands: Union[np.ndarray, torch.Tensor]
    depot_mask: Union[np.ndarray, torch.Tensor]
    graph_size: int
    depot_idx: List = [0]
    vehicle_capacity: float = -1
    max_num_vehicles: Optional[int] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def update(self, **kwargs):
        return self._replace(**kwargs)


class CCPInstance(NamedTuple):
    """Typed constrained clustering problem instance wrapper."""
    coords: Union[np.ndarray, torch.Tensor]
    demands: Union[np.ndarray, torch.Tensor]
    graph_size: int
    constraint_value: float = -1
    labels: Union[np.ndarray, torch.Tensor] = None
    num_components: Optional[int] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def update(self, **kwargs):
        return self._replace(**kwargs)
