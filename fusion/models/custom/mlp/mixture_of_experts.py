"""
fusion/models/custom/mlp/mixture_of_experts.py

Defines MixtureOfExperts: top-k routed mixture of SimpleMLP experts
with a load-balancing auxiliary loss.

Expected config parameters:
    input_dim (int)
    hidden_dims (List[int]): Hidden layer sizes used inside every expert.
    output_dim (int)
    num_experts (int): Total number of experts.
    top_k (int): Number of experts actually run per input sample.
    aux_loss_weight (float): Scale applied to the load-balancing loss,
        default 0.01.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.models.custom.mlp.simple_mlp import SimpleMLP


class MixtureOfExperts(nn.Module):
    """
    Sparse mixture of experts with top-k routing and a
    load-balancing auxiliary loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_experts: int,
        top_k: int,
        aux_loss_weight: float = 0.01,
    ):
        """
        Args:
            input_dim: Dimensionality of the input.
            hidden_dims: Hidden layer widths used inside each expert.
            output_dim: Dimensionality of the output.
            num_experts: Total number of experts available.
            top_k: Number of experts routed to per input.
                Must satisfy top_k <= num_experts.
            aux_loss_weight: Multiplier applied to the
                load-balancing loss.
        """
        super().__init__()

        assert top_k <= num_experts, (
            "top_k cannot exceed num_experts"
        )

        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.experts = nn.ModuleList(
            [
                SimpleMLP(
                    input_dim,
                    hidden_dims,
                    output_dim,
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = SimpleMLP(
            input_dim,
            [],
            num_experts,
        )

        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        batch_size = x.shape[0]

        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(
            gate_probs,
            self.top_k,
            dim=-1,
        )

        topk_probs = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True,
        )

        output = torch.zeros(
            batch_size,
            self.experts[0].net[-1].out_features,
            device=x.device,
        )

        for slot in range(self.top_k):
            expert_indices = topk_idx[:, slot]
            weights = topk_probs[:, slot].unsqueeze(-1)

            for expert_id in expert_indices.unique():
                mask = expert_indices == expert_id

                expert_out = self.experts[expert_id](x[mask])

                output[mask] += (
                    weights[mask] * expert_out
                )

        self.aux_loss = (
            self._load_balancing_loss(gate_probs)
            * self.aux_loss_weight
        )

        return output

    def _load_balancing_loss(
        self,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourages uniform expert utilization across the batch
        (Switch Transformer style).
        """
        importance = gate_probs.mean(dim=0)

        target = torch.full_like(
            importance,
            1.0 / self.num_experts,
        )

        return (
            F.mse_loss(importance, target)
            * self.num_experts
        )

    @classmethod
    def from_config(cls, config) -> "MixtureOfExperts":
        """
        Build a MixtureOfExperts from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            aux_loss_weight=config.get(
                "aux_loss_weight",
                0.01,
            ),
        )