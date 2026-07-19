import torch
import torch.nn as nn

class DenoisingAE(nn.Module):
    def __init__(
    self,
    base_ae: nn.Module,
    noise_std: float = 0.1,
) -> None:
      super().__init__()

      self.base_ae = base_ae
      self.noise_std = noise_std

      def forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
          
        """
        Forward pass through the Denoising Autoencoder.
        """

        if self.training:
         noisy_x = x + torch.randn_like(x) * self.noise_std
        else:
         noisy_x = x

        reconstruction = self.base_ae(noisy_x)

        return reconstruction
      