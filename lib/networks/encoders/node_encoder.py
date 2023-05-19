#
from warnings import warn
from typing import Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.nn as gnn

from lib.networks.encoders.graph_conv import GraphConvBlock
from lib.networks.encoders.eg_graph_conv import EGGConv
from lib.networks.encoders.base_encoder import BaseEncoder
from lib.utils import Obs, SupObs


class NodeEncoder(BaseEncoder):
    """Graph neural network encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 edge_feature_dim: int = 1,
                 num_layers: int = 3,
                 conv_type: str = "GraphConv",
                 activation: str = "gelu",
                 skip: bool = True,
                 norm_type: Optional[str] = "ln",
                 add_linear: bool = False,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            edge_feature_dim: dimension of edge features
            num_layers: number of encoder layers for neighborhood graph
            conv_type: type of graph convolution
            activation: activation function
            skip: flag to use skip (residual) connections
            norm_type: type of norm to use
            add_linear: flag to add linear layer after conv
        """
        super(NodeEncoder, self).__init__(input_dim, output_dim, hidden_dim)

        self.num_layers = num_layers
        if edge_feature_dim is not None and edge_feature_dim != 1 and conv_type.upper() != "EGGCONV":
            raise ValueError("encoders currently only work for edge_feature_dim=1")
        self.edge_feature_dim = edge_feature_dim

        self.conv_type = conv_type
        self.activation = activation
        self.skip = skip
        self.norm_type = norm_type
        self.add_linear = add_linear
        self.eggc = False

        self.input_proj = None
        self.input_proj_e = None
        self.output_proj = None
        self.layers = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self._reset_module_list(self.layers)
        if self.input_proj_e is not None:
            self.input_proj_e.reset_parameters()

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input projection layer
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        if self.conv_type.upper() == "EGGCONV":
            # special setup for EGGConv which propagates node AND edge embeddings
            self.eggc = True
            if self.activation.lower() != 'relu':
                warn(f"EGGConv normally uses RELU but got {self.activation.upper()}")
            if self.norm_type is None:
                self.norm_type = "bn"
            elif self.norm_type.lower() not in ['bn', 'batch_norm']:
                warn(f"EGGConv normally uses BN but got {self.norm_type.upper()}")
            self.input_proj_e = nn.Linear(self.edge_feature_dim, self.hidden_dim)

            def GNN():
                return EGGConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    activation=self.activation,
                    norm_type=self.norm_type,
                )
        else:
            conv = getattr(gnn, self.conv_type)

            def GNN():
                # creates a GNN module with specified parameters
                # all modules are initialized globally with the call to
                # reset_parameters()
                return GraphConvBlock(
                        conv,
                        self.hidden_dim,
                        self.hidden_dim,
                        activation=self.activation,
                        skip=self.skip,
                        norm_type=self.norm_type,
                        add_linear=self.add_linear,
                        **kwargs
                )

        # nbh based node embedding layers
        if self.num_layers > 0:
            self.layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.layers.append(GNN())

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, obs: Union[Obs, SupObs], **kwargs) -> Tensor:

        x = obs.stat_node_features
        e = obs.stat_edges
        w = obs.stat_weights
        assert e is not None and w is not None

        x = x.view(-1, x.size(-1))
        # input layer
        x = self.input_proj(x)
        # encode node embeddings
        if self.eggc:
            w = self.input_proj_e(w[:, None])
        for layer in self.layers:
            x, w = layer(x, e, w)
        # output layer
        x = self.output_proj(x)

        # check for NANs
        if self.training and (x != x).any():
            raise RuntimeError(f"Output includes NANs! (e.g. GCNConv can produce NANs when <normalize=True>!)")

        return x
