import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, in_c)
        self.decoder = Decoder(latent_dims, in_c)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        mu, logvar = self.encoder(x)
        self.kl = torch.mean(
            -0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        z = self.reparamterize(mu, logvar)
        return self.decoder(z)

    def reparamterize(self, mu, logvar):
        sigma = torch.exp(logvar / 2.0)
        eps = self.N.sample(mu.shape)
        return mu + sigma * eps
    
    def encode(self, x):
        return self.encoder(x)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 32, 5, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 5, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        # bx96x96x3 -> bx3x3x128
        self.mu_lin = nn.Linear(3 * 3 * 128, latent_dims)
        self.sigma_lin = nn.Linear(3 * 3 * 128, latent_dims)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_lin(x)
        logvar = self.sigma_lin(x)
        return [mu, logvar]


class Decoder(nn.Module):
    def __init__(self, latent_dims, in_c):
        super().__init__()
        self.linear = nn.Linear(latent_dims, 3 * 3 * 128)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 5, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_c, 6, 2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # print(f'decoder got: {z.shape}')
        z = self.linear(z)
        # print(f'after linear: {z.shape}')
        z = z.reshape(-1, 128, 3, 3)
        # print(f'after reshape: {z.shape}')
        out = self.conv_layers(z)
        # print(f'after decoder conv: {out.shape}')
        return out
