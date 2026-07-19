import torch
import torch.nn as nn
from typing import List

class SparseAE(nn.Module):
    def __init__(
    self,
    input_dim: int,
    hidden_dims: List[int],
    bottleneck_dim: int,
    lambda_sparse: float = 1e-3,
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
        self.lambda_sparse = lambda_sparse
        self.sparsity_loss = torch.tensor(0.0)

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
            Decode the bottleneck representation back to the original input space.
          """
          return self.decoder(z)
        
        def forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
           
         """
          Forward pass through the Sparse Autoencoder.
         """
         bottleneck = self.encode(x)

         self.sparsity_loss = (
         self.lambda_sparse *
         bottleneck.abs().mean()
      )

         reconstruction = self.decode(bottleneck)

         return reconstruction