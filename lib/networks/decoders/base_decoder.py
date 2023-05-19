#
from abc import abstractmethod
from typing import Tuple, Optional
from torch import Tensor, BoolTensor
import torch.nn as nn
from torch.distributions.categorical import Categorical


class BaseDecoder(nn.Module):
    """Abstract decoder model."""
    def __init__(self,
                 center_emb_dim: int,
                 node_emb_dim: int,
                 hidden_dim: int = 128,
                 **kwargs):
        """

        Args:
            query_emb_dim: dimension of query embedding
            action_emb_dim: dimension of action embedding
            hidden_dim: dimension of hidden layers
            decode_type: decoding strategy, ['greedy', 'sampling']
        """
        super(BaseDecoder, self).__init__()
        self.center_emb_dim = center_emb_dim
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def create_layers(self, **kwargs):
        """Create the specific model layers."""
        raise NotImplementedError

    def reset_parameters(self):
        pass

    def forward(self,
                center_emb: Tensor,
                node_emb: Tensor,
                mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """
        Model specific implementation of forward pass.

        Args:
            center_emb: (BS, K, center_emb_dim)
            node_emb: (BS, N, node_emb_dim)
            mask: (BS, N)

        Returns:
            action, log_likelihood, entropy

        """
        raise NotImplementedError


class RLBaseDecoder(nn.Module):
    """Abstract decoder model."""
    def __init__(self,
                 query_emb_dim: int,
                 action_emb_dim: int,
                 hidden_dim: int = 128,
                 decode_type: str = "greedy",
                 **kwargs):
        """

        Args:
            query_emb_dim: dimension of query embedding
            action_emb_dim: dimension of action embedding
            hidden_dim: dimension of hidden layers
            decode_type: decoding strategy, ['greedy', 'sampling']
        """
        super(RLBaseDecoder, self).__init__()
        self.query_emb_dim = query_emb_dim
        self.action_emb_dim = action_emb_dim
        self.hidden_dim = hidden_dim
        self.decode_type = decode_type

    @abstractmethod
    def create_layers(self, **kwargs):
        """Create the specific model layers."""
        raise NotImplementedError

    def reset_parameters(self):
        pass

    def forward(self,
                env,
                query_emb: Tensor,
                action_emb: Tensor,
                mask: BoolTensor,
                **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Model specific implementation of forward pass.

        Args:
            env: environment object
            query_emb: (BS, query_emb_dim) query embedding
            action_emb: (BS, num_a, action_emb_dim) action embedding
            mask: (BS, num_a) mask indicating infeasible action indices over action set

        Returns:
            action, log_likelihood, entropy

        """
        raise NotImplementedError

    def set_decode_type(self, decode_type: str) -> None:
        """
        Set the decoding type:
            - 'greedy'
            - 'sampling'

        Args:
            decode_type (str): type of decoding

        """
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def _select_a(self,
                  logits: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select an action given the logits from the decoder."""

        assert (logits == logits).all(), "Logits contain NANs!"
        ent = None
        if self.decode_type == "greedy":
            a_idx = logits.argmax(dim=-1)
        elif self.decode_type == "sampling":
            #dist = ActionDist(logits=logits)
            dist = Categorical(logits=logits)
            a_idx = dist.sample()
            ent = dist.entropy()
        else:
            raise RuntimeError(f"Unknown decoding type <{self.decode_type}> (Forgot to 'set_decode_type()' ?)")

        # get corresponding log likelihood
        ll = logits.gather(-1, a_idx.unsqueeze(-1))
        assert (ll > -1000).data.all(), "Log_probs are -inf, check sampling procedure!"
        return a_idx, ll, ent


class ActionDist(Categorical):
    """
    Adapted categorical distribution that
    ignores -inf masks when calculating entropy.
    """

    def entropy(self) -> Tensor:
        """Ignores -inf in calculation"""
        non_inf_idx = self.logits >= -10000
        p_log_p = self.logits[non_inf_idx] * self.probs[non_inf_idx]
        return -p_log_p.sum(-1)

