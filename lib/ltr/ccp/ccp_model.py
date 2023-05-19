#
import logging
from typing import Optional, Dict, Tuple
from torch import Tensor
import torch.nn as nn

from lib.utils import count_parameters, SupObs
from lib.networks.encoders import NodeEncoder, CCPCenterEncoder
import lib.networks.decoders as decoders

logger = logging.getLogger(__name__)


class CCPModel(nn.Module):
    """
    Model wrapping encoder and decoder models.

    Args:
        input_dim: input dimension of nodes
        decoder_type: type of decoder to use
        node_encoder_args: additional arguments for encoder creation
        center_encoder_args:
        decoder_args: additional arguments for decoder creation
        embedding_dim: general embedding dimension of model

    """

    def __init__(self,
                 input_dim: int,
                 decoder_type: str = "PointwiseDecoder",
                 node_encoder_args: Optional[Dict] = None,
                 center_encoder_args: Optional[Dict] = None,
                 decoder_args: Optional[Dict] = None,
                 embedding_dim: int = 256,
                 **kwargs):
        super(CCPModel, self).__init__()

        self.input_dim = input_dim
        self.decoder_type = decoder_type
        self.node_encoder_args = node_encoder_args if node_encoder_args is not None else {}
        self.center_encoder_args = center_encoder_args if center_encoder_args is not None else {}
        self.decoder_args = decoder_args if decoder_args is not None else {}
        self.embedding_dim = embedding_dim

        # initialize encoder models
        self.node_encoder = NodeEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim,
            edge_feature_dim=1,
            **self.node_encoder_args, **kwargs
        )

        self.center_encoder = CCPCenterEncoder(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            **self.center_encoder_args, **kwargs
        )

        # initialize decoder model
        d_cl = getattr(decoders, decoder_type)
        self.decoder = d_cl(
            center_emb_dim=embedding_dim,
            node_emb_dim=embedding_dim,
            **self.decoder_args, **kwargs
        )

        self.reset_parameters()

    def __repr__(self):
        super_repr = super().__repr__()  # get torch module str repr
        n_enc_p = count_parameters(self.node_encoder)
        c_enc_p = count_parameters(self.center_encoder)
        dec_p = count_parameters(self.decoder)
        add_repr = f"\n-----------------------------------" \
                   f"\nNum Parameters: " \
                   f"\n  (node_encoder): {n_enc_p} " \
                   f"\n  (center_encoder): {c_enc_p} " \
                   f"\n  (decoder): {dec_p} " \
                   f"\n  total: {n_enc_p + c_enc_p + dec_p}\n"
        return super_repr + add_repr

    def reset_parameters(self):
        """Reset model parameters."""
        self.node_encoder.reset_parameters()
        self.center_encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self,
                nodes: Tensor,
                centroids: Tensor,
                medoids: Tensor,
                c_mask: Optional[Tensor] = None,
                edges: Optional[Tensor] = None,
                weights: Optional[Tensor] = None,
                node_emb: Optional[Tensor] = None,
                assignment: Optional[Tensor] = None,
                n_mask: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            nodes: (BS, N, D)
            centroids: (BS, K, 2)
            medoids: (BS, K)
            c_mask: (BS, K)     # masks dummy centers
            edges: (2, BSM, )
            weights: (BSM, )
            node_emb: (BS, N, emb_dim)
            assignment: (BS, N, )
            n_mask: (BS, N, )   # masks nodes

        Returns:

        """
        bs, n, d = nodes.size()
        assert len(medoids.shape) == 2
        assert len(centroids.shape) == 3
        assert medoids.size(0) == centroids.size(0) == bs

        if node_emb is None:
            # run node encoder to create node embeddings
            obs = SupObs(
                stat_node_features=nodes,
                stat_edges=edges,   # type: ignore
                stat_weights=weights,
            )
            node_emb = self.node_encoder(obs).view(bs, -1, self.embedding_dim)
        else:
            assert node_emb.size(0) == bs and \
                   node_emb.size(1) == n and \
                   node_emb.size(2) == self.embedding_dim

        # encode centers
        center_emb = self.center_encoder(
            centroids, medoids, node_emb, c_mask, assignment
        )

        scores = self.decoder(
            center_emb=center_emb,
            node_emb=node_emb,
            mask=n_mask,
        )
        return scores, node_emb
