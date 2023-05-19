#
from typing import Union, Optional
import torch
from torch import Tensor

from lib.networks.utils import NN
from lib.networks.decoders.base_decoder import BaseDecoder

COMB_MODES = ["cat", "sum"]


class PointwiseDecoder(BaseDecoder):
    """Simple FF pointwise scoring function."""

    def __init__(self,
                 center_emb_dim: int,
                 node_emb_dim: int,
                 hidden_dim: int = 128,
                 comb_mode: str = "cat",
                 num_layers: int = 2,
                 activation: str = "gelu",
                 norm_type: Optional[str] = "ln",
                 **kwargs):
        super(PointwiseDecoder, self).__init__(
            center_emb_dim=center_emb_dim,
            node_emb_dim=node_emb_dim,
            hidden_dim=hidden_dim
        )
        self.comb_mode = comb_mode.lower()
        self.num_layers = num_layers
        self.activation = activation
        self.norm_type = norm_type.lower() if norm_type is not None else None
        assert self.comb_mode in COMB_MODES
        assert center_emb_dim == node_emb_dim

        self.nn = None
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        dim = 2*self.node_emb_dim if self.comb_mode == "cat" else self.node_emb_dim
        self.nn = NN(
            in_=dim,
            out_=1,
            h_=self.hidden_dim,
            nl=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            **kwargs
        )

    def reset_parameters(self):
        self._reset_module_list(self.nn)

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self,
                center_emb: Tensor,
                node_emb: Tensor,
                mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """

        Args:
            center_emb: (BS, k_max, D1)
            node_emb: (BS, N, D2)
            mask: (BS, N, )
            **kwargs:

        Returns:
            scores: (BS, k_max, N, 1)
        """
        k_max = center_emb.size(1)
        n = node_emb.size(1)
        center_emb = center_emb[:, :, None, :].expand(-1, -1, n, -1)
        node_emb = node_emb[:, None, :, :].expand(-1, k_max, -1, -1)

        if self.comb_mode == "cat":
            emb = torch.cat((center_emb, node_emb), dim=-1)
        elif self.comb_mode == "sum":
            emb = center_emb + node_emb
        else:
            raise ValueError(self.comb_mode)

        return self.nn(emb).squeeze(-1)

