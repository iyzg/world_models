# TODO: RNN model
import torch
from torch import nn
from torch.nn import functional as F

class MDN(nn.Module):
    def __init__(self, input_size, output_size, n_gaussians, temp=1.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_gaussians = n_gaussians

        self.pi_lin = nn.Linear(input_size, n_gaussians)
        self.mu_lin = nn.Linear(input_size, n_gaussians * output_size)
        self.sigma_lin = nn.Linear(input_size, n_gaussians * output_size)

        self.temp = temp

    def forward(self, x):
        # Drop down to one timestep at a time
        x = x.squeeze(0)
        pi = F.softmax(self.pi_lin(x), dim=1)

        mu = self.mu_lin(x)
        mu = mu.reshape(-1, self.n_gaussians, self.output_size)

        sigma = torch.exp(self.sigma_lin(x))
        sigma = sigma.reshape(-1, self.n_gaussians, self.output_size)

        return pi, mu, sigma

    def loss(self, pi, mu, sigma, y):
        n = torch.distributions.Normal(mu, sigma)
        y = y.unsqueeze(1).expand_as(mu)
        logprob = n.log_prob(y)
        logprob = logprob.sum(-1)   # [b, n_gau]

        # log(pi * prob) = log(pi) + log(prob)
        log_pi = torch.log(pi + 1e-6)
        logprob = logprob + log_pi

        prob = torch.logsumexp(logprob, dim=1)
        return -prob.mean()

class MDNRNN(nn.Module):
    def __init__(self, n_gaussians, input_size, hidden_size, output_size, layers):
        super().__init__()

        self.layers = layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size, hidden_size, layers, batch_first=True, proj_size=output_size)
        self.mdn = MDN(output_size, output_size, n_gaussians)

    def forward(self, x, h):
        y, h = self.rnn(x, h)
        pi, mu, sigma = self.mdn(y)
        return pi, mu, sigma, h

    def initial_state(self, batch_size):
        return (torch.zeros(self.layers, batch_size, self.output_size),
                torch.zeros(self.layers, batch_size, self.hidden_size))
