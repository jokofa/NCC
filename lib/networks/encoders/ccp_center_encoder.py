#
from typing import Tuple, List, Optional, Dict, Any, Union
from warnings import warn
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.aggr.gmt import SAB

from lib.utils import rm_from_kwargs
from lib.networks.utils import NN
from lib.networks.encoders.base_encoder import BaseEncoder, COMB_TYPES


class CCPCenterEncoder(BaseEncoder):
    """Encoder model for center context embeddings."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 pooling_type: Union[str, List[str], Tuple[str, str]] = ("mean", "max"),
                 meta_comb_mode: str = "cat",
                 pooling_args: Optional[Dict[str, Any]] = None,
                 pre_proj: bool = True,
                 post_proj: bool = True,
                 attn: bool = True,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 activation: str = "gelu",
                 norm_type: Optional[str] = "ln",
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            num_layers: number of hidden layers

        """
        super(CCPCenterEncoder, self).__init__(input_dim, output_dim, hidden_dim)
        self.pooling_type = [pooling_type] if isinstance(pooling_type, str) else list(pooling_type)
        for pt in self.pooling_type:
            assert pt in COMB_TYPES
        self.meta_comb_mode = meta_comb_mode.lower()
        if self.meta_comb_mode not in COMB_TYPES:
            raise ValueError(f"unsupported meta_comb_mode: '{self.meta_comb_mode}'")
        self.pooling_args = pooling_args if pooling_args is not None else {}
        self.pre_proj = pre_proj
        self.post_proj = post_proj
        self.attn = attn
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.norm_type = norm_type

        self._comb = None
        self.pool_opt = None
        #self.centroid_net = None
        self.center_net = None
        self.graph_net = None
        self.comb_net = None
        self.pre_net = None
        self.post_net = None
        self.attn_net = None

        kwargs = rm_from_kwargs(kwargs, keys=["edge_feature_dim"])
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # global pooling operator to pool over graph
        npool = len(self.pooling_type)
        self.pool_opt = [getattr(torch, pt) for pt in self.pooling_type]

        self.center_net = NN(
            in_=self.hidden_dim,
            out_=self.hidden_dim,
            h_=self.hidden_dim,
            nl=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )

        self.graph_net = NN(
            in_=(npool * self.hidden_dim),
            out_=self.hidden_dim,
            h_=self.hidden_dim,
            nl=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )

        if self.meta_comb_mode == "cat":
            pass
        elif self.meta_comb_mode == "proj":
            self.comb_net = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        else:
            self._comb = getattr(torch, self.meta_comb_mode)

        if self.pre_proj:
            self.pre_net = NN(
                in_=self.input_dim,
                out_=self.hidden_dim,
                h_=self.hidden_dim,
                nl=1,
                activation=self.activation,
                norm_type=None,
            )
        else:
            assert self.input_dim == self.hidden_dim

        if self.attn:
            dim = 2 * self.hidden_dim if self.meta_comb_mode == "cat" else self.hidden_dim
            self.attn_net = SAB(
                in_channels=dim,
                out_channels=self.hidden_dim,
                num_heads=self.num_heads,
                layer_norm=(self.norm_type is not None and self.norm_type == "ln")
            )

        if self.post_proj:
            dim = 2 * self.hidden_dim if (self.meta_comb_mode == "cat" and not self.attn) else self.hidden_dim
            self.post_net = NN(
                in_=dim,
                out_=self.output_dim,
                h_=self.hidden_dim,
                nl=self.num_layers,
                activation=self.activation,
                norm_type=None,
            )
        else:
            assert self.output_dim == self.hidden_dim

    def pool(self, x: torch.Tensor, dim=-1) -> List[torch.Tensor]:
        tmp = []
        for pool in self.pool_opt:
            out = pool(x, dim=dim)
            tmp.append(out if isinstance(out, Tensor) else out[0])
        return tmp

    def _get_center_emb(self,
                        #centroids: Tensor,
                        medoids: Tensor,
                        node_emb: Tensor) -> Tensor:
        """

        Args:
            centroids: (BS, k_max, 2)
            medoids: (BS, k_max)
            node_emb: (BS, N, D)

        Returns:

        """
        bs, n, d = node_emb.size()
        k_max = medoids.size(-1)
        return self.center_net(
            node_emb.gather(
                index=medoids[:, :, None].expand(-1, -1, d),
                dim=1
            ).view(bs, k_max, d)
        )

    def _get_graph_emb(self, node_emb: Tensor):
        pooled_emb = self.pool(node_emb, dim=1)
        if len(pooled_emb) > 1:
            pooled_emb = torch.cat(pooled_emb, dim=-1)
        else:
            pooled_emb = pooled_emb[0]
        return self.graph_net(pooled_emb)

    def _merge_emb(self, c_emb: torch.Tensor, g_emb: torch.Tensor):
        bs, k, d = c_emb.size()
        g_emb = g_emb[:, None, :].expand(-1, k, -1)
        # combine embeddings
        if self.meta_comb_mode == "cat":
            return torch.cat((c_emb, g_emb), dim=-1)
        elif self.meta_comb_mode == "proj":
            return self.comb_net(torch.cat((c_emb, g_emb), dim=-1))
        else:
            out = self._comb(torch.stack((c_emb, g_emb), dim=0), dim=0)
            return out if isinstance(out, Tensor) else out[0]

    def forward(self,
                centroids: torch.Tensor,
                medoids: torch.Tensor,
                node_emb: torch.Tensor,
                c_mask: Optional[Tensor] = None,
                assignment: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        if assignment is not None:
            raise NotImplementedError()

        # input proj
        if self.pre_net is not None:
            node_emb = self.pre_net(node_emb)

        c_emb = self._get_center_emb(medoids, node_emb)
        g_emb = self._get_graph_emb(node_emb)
        emb = self._merge_emb(c_emb, g_emb)

        # self-attention
        if self.attn:
            msk = None
            if c_mask is not None:
                # mask dummy centers for attention
                msk = torch.zeros_like(c_mask, dtype=emb.dtype, device=emb.device)
                msk[c_mask] = -float("inf")     # mask is additive!
                msk = msk.unsqueeze(-1)
            emb = self.attn_net(x=emb, mask=msk)

        # output proj
        if self.post_net is not None:
            emb = self.post_net(emb)

        return emb
