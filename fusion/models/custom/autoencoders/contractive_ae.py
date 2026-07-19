import torch
import torch.nn as nn
from typing import List

class ContractiveAE(nn.Module):
    def __init__(
    self,
    input_dim: int,
    hidden_dims: List[int],
    bottleneck_dim: int,
    lambda_contractive: float = 1e-3,
) -> None:
        super().__init__()
        encoder_layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
           encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
           encoder_layers.append(nn.ReLU())
           prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []

        prev_dim = bottleneck_dim

        for hidden_dim in reversed(hidden_dims):
           decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
           decoder_layers.append(nn.ReLU())
           prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        self.lambda_contractive = lambda_contractive
        self.contraction_loss = torch.tensor(0.0)

        def encode(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
         """
          Encode the input into a bottleneck representation.
         """
         return self.encoder(x)
        
        def decode(
    self,
    z: torch.Tensor,
) -> torch.Tensor:
         """
         Decode the bottleneck representation.
        """
         return self.decoder(z)
        
        def forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
         """
          Forward pass through the Contractive Autoencoder.
         """

         bottleneck = self.encode(x)
         h = torch.sigmoid(bottleneck)
         weight_norm = self.encoder[-1].weight.pow(2).sum()

         contractive_penalty = (
    h * (1 - h)
).sum() * weight_norm
         
         self.contraction_loss = (
    self.lambda_contractive *
    contractive_penalty
)
         reconstruction = self.decode(bottleneck)

         return reconstruction
