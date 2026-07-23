"""
fusion/models/custom/mlp/adaptive_mlp.py

Defines AdaptiveMLP: an early-exit MLP whose effective depth varies
per input.

Expected config parameters:
    input_dim (int)
    hidden_dim (int): Width shared by every hidden layer.
    num_layers (int): Total number of hidden layers
        (equals number of exit points).
    output_dim (int)
    exit_threshold (float): Confidence required to exit early
        at inference time.
    activation (str): Default "relu".
"""

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation


class AdaptiveMLP(nn.Module):
    """
    MLP with one early-exit head per hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        exit_threshold: float = 0.9,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Width shared by every hidden layer.
            num_layers: Number of hidden layers
                (and exit points).
            output_dim: Dimensionality of each exit prediction.
            exit_threshold: Confidence score above which
                inference exits early.
            activation: Activation used after each hidden layer.
        """
        super().__init__()

        self.exit_threshold = exit_threshold

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    get_activation(activation),
                )
                for _ in range(num_layers)
            ]
        )

        self.exit_heads = nn.ModuleList(
            [
                nn.Linear(hidden_dim, output_dim)
                for _ in range(num_layers)
            ]
        )

        self.exit_gates = nn.ModuleList(
            [
                nn.Linear(hidden_dim, 1)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            If training:
                List of (prediction, confidence) tuples,
                one per exit.

            If evaluating:
                Tensor of shape (B, output_dim) from the
                earliest confident exit. If no exit meets
                the threshold, returns the final exit output.
        """
        h = self.input_proj(x)

        if self.training:
            outputs = []

            for layer, head, gate in zip(
                self.hidden_layers,
                self.exit_heads,
                self.exit_gates,
            ):
                h = layer(h)

                prediction = head(h)
                confidence = torch.sigmoid(gate(h))

                outputs.append(
                    (prediction, confidence)
                )

            return outputs

        for layer, head, gate in zip(
            self.hidden_layers,
            self.exit_heads,
            self.exit_gates,
        ):
            h = layer(h)

            confidence = torch.sigmoid(gate(h))

            if confidence.mean().item() > self.exit_threshold:
                return head(h)

        # No exit was confident enough.
        return self.exit_heads[-1](h)

    @staticmethod
    def compute_loss(
        outputs,
        target,
        loss_fn,
    ) -> torch.Tensor:
        """
        Sum the losses from all exit heads.

        Args:
            outputs: List returned by forward() in training mode.
            target: Ground-truth labels.
            loss_fn: Loss function such as
                nn.CrossEntropyLoss() or nn.MSELoss().

        Returns:
            Scalar tensor containing the total loss.
        """
        total = torch.tensor(0.0)

        for prediction, _confidence in outputs:
            total = total + loss_fn(
                prediction,
                target,
            )

        return total

    @classmethod
    def from_config(cls, config) -> "AdaptiveMLP":
        """
        Build an AdaptiveMLP from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            exit_threshold=config.get(
                "exit_threshold",
                0.9,
            ),
            activation=config.get(
                "activation",
                "relu",
            ),
        )