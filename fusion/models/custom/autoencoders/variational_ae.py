import torch
import torch.nn as nn
from typing import List

class VariationalAE(nn.Module):
    """
    Variational Autoencoder implemented using Multi-Layer Perceptrons.
    """

    def __init__(
    self,
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
)-> None:
        super().__init__()
        encoder_layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
          encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
          encoder_layers.append(nn.ReLU())
          prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []

        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
           decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
           decoder_layers.append(nn.ReLU())
           prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_loss_value = torch.tensor(0.0, dtype=torch.float32)

def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

def encode(
    self,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode the input into mean and log variance.
    """
    hidden = self.encoder(x)

    mu = self.fc_mu(hidden)
    log_var = self.fc_log_var(hidden)

    return mu, log_var

def decode(
    self,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Decode a latent vector into a reconstruction.
    """
    return self.decoder(z)
def forward(
    self,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass through the Variational Autoencoder.
    """
    mu, log_var = self.encode(x)

    z = self.reparameterize(mu, log_var)

    reconstruction = self.decode(z)

    self.kl_loss_value = self.kl_loss(mu, log_var)

    return reconstruction, mu, log_var

def kl_loss(
    self,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence loss.
    """
    return -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - torch.exp(log_var)
    )
        
        