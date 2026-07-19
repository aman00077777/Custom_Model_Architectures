import torch
import torch.nn as nn
from typing import List

class VanillaAE(nn.Module):
    """
    Vanilla Autoencoder implemented using Multi-Layer Perceptrons.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
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

        def encode(self, x: torch.Tensor) -> torch.Tensor:
           """
            Encode the input into the latent representation.
           """
           return self.encoder(x)
        
        def decode(self, z: torch.Tensor) -> torch.Tensor:
           """
            Decode the latent representation back to the original input space.
           """
           return self.decoder(z)
        
        def forward(
    self,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
               """
                Forward pass through the autoencoder.

                Returns:
                bottleneck: Latent representation.
                reconstruction: Reconstructed input.
              """
               bottleneck = self.encode(x)
               reconstruction = self.decode(bottleneck)

               return bottleneck, reconstruction
