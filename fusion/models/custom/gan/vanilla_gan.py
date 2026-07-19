import torch
import torch.nn as nn
from typing import List

class VanillaGAN(nn.Module):
    def __init__(
    self,
    latent_dim: int,
    hidden_dims: List[int],
    output_dim: int,
) -> None:
        super().__init__()
        generator_layers = []

        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            generator_layers.append(nn.Linear(prev_dim, hidden_dim))
            generator_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        generator_layers.append(nn.Linear(prev_dim, output_dim))
        generator_layers.append(nn.Tanh())

        self.generator = nn.Sequential(*generator_layers)
        
        discriminator_layers = []

        prev_dim = output_dim

        for hidden_dim in hidden_dims:
            discriminator_layers.append(nn.Linear(prev_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        discriminator_layers.append(nn.Linear(prev_dim, 1))
        discriminator_layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*discriminator_layers)

    def generate(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate fake samples from latent vectors.
        """
        return self.generator(z)
        
    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the generator.
        """
        return self.generate(z)
        
    def generator_loss(
        self,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the generator loss.
        """
        predictions = self.discriminator(fake_samples)

        targets = torch.ones_like(predictions)

        loss = nn.functional.binary_cross_entropy(
           predictions,
           targets,
        )

        return loss
        
    def discriminator_loss(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the discriminator loss.
        """
        real_predictions = self.discriminator(real_samples)
        fake_predictions = self.discriminator(fake_samples)

        real_targets = torch.ones_like(real_predictions)
        fake_targets = torch.zeros_like(fake_predictions)

        real_loss = nn.functional.binary_cross_entropy(
        real_predictions,
        real_targets,
        )

        fake_loss = nn.functional.binary_cross_entropy(
        fake_predictions,
        fake_targets,
        )

        return (real_loss + fake_loss) / 2
