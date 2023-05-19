#
from typing import Dict, Optional, List
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str, module: bool = False, negative_slope: float = 0.01, **kwargs):
    if activation is None:
        return None
    if activation.upper() == "RELU":
        return F.relu if not module else nn.ReLU(**kwargs)
    elif activation.upper() == "GELU":
        return F.gelu if not module else nn.GELU()
    elif activation.upper() == "TANH":
        return F.tanh if not module else nn.Tanh()
    elif activation.upper() == "LEAKYRELU":
        return F.leaky_relu if not module else nn.LeakyReLU(negative_slope, **kwargs)
    else:
        raise ModuleNotFoundError(activation)


def get_norm(norm_type: str, hdim: int, **kwargs):
    if norm_type is None:
        return None
    if norm_type.lower() in ['bn', 'batch_norm']:
        return nn.BatchNorm1d(hdim, **kwargs)
    elif norm_type.lower() in ['ln', 'layer_norm']:
        return nn.LayerNorm(hdim, **kwargs)
    else:
        raise ModuleNotFoundError(norm_type)


def NN(in_: int, out_: int, h_: int, nl: int = 1, 
       activation: str = "relu", 
       norm_type: Optional[str] = None, 
       **kwargs):
    """Creates a FF neural net.
    Layer ordering: (Lin -> Act -> Norm)
    """
    if nl == 1:
        return nn.Linear(in_, out_)
    elif nl == 2:
        layers = [nn.Linear(in_, h_)]
        layers.append(get_activation_fn(activation, module=True, **kwargs))
        if norm_type is not None:
            layers.append(get_norm(norm_type, hdim=h_, **kwargs))
        layers.append(nn.Linear(h_, out_))
    else:
        layers = [nn.Linear(in_, h_)]
        for _ in range(max(nl - 2, 0)):
            layers.append(get_activation_fn(activation, module=True, **kwargs))
            if norm_type is not None:
                layers.append(get_norm(norm_type, hdim=h_, **kwargs))
            layers.append(nn.Linear(h_, h_))
        layers.append(get_activation_fn(activation, module=True, **kwargs))
        if norm_type is not None:
            layers.append(get_norm(norm_type, hdim=h_, **kwargs))
        layers.append(nn.Linear(h_, out_))
    return nn.Sequential(*layers)
