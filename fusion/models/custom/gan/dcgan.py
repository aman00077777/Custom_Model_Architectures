import torch
import torch.nn as nn

class DCGAN(nn.Module):
    def __init__(
    self,
    latent_dim: int = 100,
    image_channels: int = 3,
    feature_maps: int = 64,
) -> None:
        super().__init__()
        self.generator = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(feature_maps * 8),
    nn.ReLU(True),

    nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 4),
    nn.ReLU(True),

    nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 2),
    nn.ReLU(True),

    nn.ConvTranspose2d(feature_maps * 2, image_channels, 4, 2, 1, bias=False),
    nn.Tanh(),
)
        
        self.discriminator = nn.Sequential(
    nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 2),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 4),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
    nn.Sigmoid(),
)

    def generate(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate fake images from latent vectors.
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
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute generator loss.
        """
        predictions = self.discriminator(fake_images)

        targets = torch.ones_like(predictions)

        return nn.functional.binary_cross_entropy(
            predictions,
            targets,
        )
    
    def discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        """
        real_predictions = self.discriminator(real_images)
        fake_predictions = self.discriminator(fake_images)

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
