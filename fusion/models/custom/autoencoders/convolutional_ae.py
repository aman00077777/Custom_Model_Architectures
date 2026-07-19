import torch
import torch.nn as nn

class ConvolutionalAE(nn.Module):
    """
    Convolutional Autoencoder implemented using Convolutional Neural Networks.
    """

    def __init__(self):
     super().__init__()
     self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
)
     self.decoder = nn.Sequential(
    nn.ConvTranspose2d(
        64,
        32,
        kernel_size=2,
        stride=2,
    ),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.ConvTranspose2d(
        32,
        3,
        kernel_size=2,
        stride=2,
    ),
    nn.Sigmoid(),
)
     
def encode(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Encode the input image.
    """
    return self.encoder(x)

def decode(
    self,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Decode the latent representation.
    """
    return self.decoder(z)

def forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass through the Convolutional Autoencoder.
    """
    z = self.encode(x)
    reconstruction = self.decode(z)
    return reconstruction
