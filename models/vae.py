import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, in_c)
        self.decoder = Decoder(latent_dims, in_c)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 16, 5, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2),
            nn.ReLU(),
        )

        # bx96x96x3 -> bx3x3x128
        self.mu_lin = nn.Linear(3*3*128, latent_dims)
        self.sigma_lin = nn.Linear(3*3*128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        # print(f'shape before conv: {x.shape}')
        x = self.conv_layers(x)
        # print(f'shape after conv: {x.shape}')
        x = torch.flatten(x, start_dim=1)
        # print(f'shape after flatten: {x.shape}')
        mu = self.mu_lin(x)
        sigma = torch.exp(self.sigma_lin(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # print(f'after: {z.shape}')
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.linear = nn.Linear(latent_dims, 3*3*128)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_c, 6, 2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # print(f'decoder got: {z.shape}')
        z = self.linear(z)
        z = F.relu(z)
        # print(f'after linear: {z.shape}')
        z = z.reshape(-1, 128, 3, 3)
        # print(f'after reshape: {z.shape}')
        out = self.conv_layers(z)
        # print(f'after decoder conv: {out.shape}')
        return out