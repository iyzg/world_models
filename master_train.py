# Normal comments
## Teaching comments

import bisect
import glob
import math
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------
# VAE building blocks

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, bias = False)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = NewGELU()
    
    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class VariationalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [config.in_channels] + list(config.conv_channels)
        self.conv_layers = nn.ModuleList([
            EncoderBlock(channels[i], channels[i+1], config.enc_kernel_sizes[i], config.stride)
            for i in range(len(channels) - 1)
        ])

        self.mu = nn.Linear(config.flattened_size, config.latent_dims)
        self.logvar = nn.Linear(config.flattened_size, config.latent_dims)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return [mu, logvar]

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, bias = False)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = NewGELU()
    
    def forward(self, x):
        x = self.act(self.norm(self.deconv(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = list(config.conv_channels[::-1]) + [config.in_channels] 
        self.deconv_layers = nn.ModuleList([
            DecoderBlock(self.channels[i], self.channels[i+1], config.dec_kernel_sizes[i], config.stride)
            for i in range(len(self.channels) - 1)
        ])
        # Change last layer to sigmoid
        self.deconv_layers[-1].act = nn.Sigmoid()
        self.mlp = nn.Linear(config.latent_dims, config.flattened_size)
        self.h, self.w = config.compressed_size

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, self.channels[0], self.h, self.w)
        for layer in self.deconv_layers:
            x = layer(x)
        return x

# --------------------------------------------------------------------
# Main VAE model

# NOTE: This is a dataclass for future if you want different configurations
@dataclass
class VAEConfig:
    input_size: tuple = (96, 96)
    in_channels: int = 3

    latent_dims: int = 32

    conv_channels: tuple = (32, 64, 128, 256)  # Channel sizes for conv layers
    enc_kernel_sizes: tuple = (5, 5, 5, 5)
    dec_kernel_sizes: tuple = (5, 5, 6, 6)
    stride: int = 2  # Stride for conv/deconv layers

    def __post_init__(self):
        # Calculate the size of the flattened representation after convolutions
        # This is useful for creating the linear layers between conv and latent space
        h, w = self.input_size
        for kernel in self.enc_kernel_sizes:
            h = (h - (kernel - 2)) // self.stride
            w = (w - (kernel - 2)) // self.stride

        self.compressed_size = (h, w)
        self.flattened_size = h * w * self.conv_channels[-1]

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VariationalEncoder(config)
        self.decoder = Decoder(config)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        # TODO: How to do device here correctly for N

    def forward(self, x):
        mu, logvar = self.encoder(x)
        self.kl = torch.mean(
            -0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        z = self.reparamterize(mu, logvar)
        return self.decoder(z)

    def reparamterize(self, mu, logvar):
        sigma = torch.exp(logvar / 2.0)
        eps = self.N.sample(mu.shape).to(mu.device)
        return mu + sigma * eps
    
    def encode(self, x):
        return self.encoder(x)

# --------------------------------------------------------------------
# Main RNN model

@dataclass
class RNNConfig:
    input_size: int = 32 + 3    # latent_dims + action_size
    hidden_size: int = 128
    output_size: int = 32
    n_gaussians: int = 5
    n_layers: int = 1
    temp: float = 1.0

# TODO: TOTALLY GO THROUGH AND EVICERATE THIS STUFF
class MDN(nn.Module):
    # def __init__(self, input_size, output_size, n_gaussians, temp=1.0):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.n_gaussians = config.n_gaussians 

        self.pi_lin = nn.Linear(config.input_size, config.n_gaussians)
        self.mu_lin = nn.Linear(config.input_size, config.n_gaussians * config.output_size)
        self.sigma_lin = nn.Linear(config.input_size, config.n_gaussians * config.output_size)

        self.temp = config.temp 

    def forward(self, x):
        # Drop down to one timestep at a time
        x = x.squeeze(0)
        pi = F.softmax(self.pi_lin(x), dim=1)

        mu = self.mu_lin(x)
        mu = mu.reshape(-1, self.n_gaussians, self.output_size)

        sigma = torch.exp(self.sigma_lin(x))
        sigma = sigma.reshape(-1, self.n_gaussians, self.output_size)

        return pi, mu, sigma

class MDNRNN(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x, h):
        y, h = self.rnn(x, h)
        pi, mu, sigma = self.mdn(y)
        return pi, mu, sigma, h


    def loss(self, pi, mu, sigma, y):
        n = torch.distributions.Normal(mu, sigma)
        y = y.unsqueeze(1).expand_as(mu)
        logprob = n.log_prob(y)
        logprob = logprob.sum(-1)   # [b, n_gau]

        ## log(a * b) = log(a) + log(b)
        ## this is a classic trick of using log to turn a product into a sum
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

# --------------------------------------------------------------------
# Dataset classes

class FrameDataset(Dataset):
    def __init__(self, file_pattern):
        self.file_list = sorted(glob.glob(file_pattern))
        # Store mapping from flat index to (file_idx, frame_idx)
        self.file_sizes = []  # Size of each file in frames
        self.cumulative_sizes = [0]  # Cumulative sum of frames
        
        for file in self.file_list:
            # Just read the header to get array size
            array = np.load(file, mmap_mode="r")["obs"]
            # Assuming shape is (num_frames, frame_height, frame_width, channels)
            num_frames = array.shape[0]
            self.file_sizes.append(num_frames)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_frames)

    def __len__(self):
        return self.cumulative_sizes[-1]  # Total number of frames
        
    def __getitem__(self, idx):
        # Binary search to find which file contains this frame
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        frame_idx = idx - self.cumulative_sizes[file_idx]
        
        # Memory map the file and get the specific frame
        array = np.load(self.file_list[file_idx], mmap_mode='r')["obs"]
        return torch.from_numpy(array[frame_idx].copy().astype(np.float32) / 255.0)


# ---------------------------------------------------------------------
# Utils
def get_device(args):
    if args.device:
        # provided explicitly by the user
        device = args.device
    else:
        # attempt to autodetect the device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    return device

# ---------------------------------------------------------------------
# Rollout extraction
def rollout(worker_id, trials, args):
    # TODO: Change env to be instantiated from parameter
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    total_frames = 0

    for trial in range(trials):
        seed = random.randint(0, 2**31 - 1)
        filename = f'{args.rollout_dir}/{worker_id}_{seed}.npz'
        obs_l = []
        act_l = []

        # obs is [96, 96, 3]
        obs, info = env.reset(seed=seed)
        for frame in range(args.max_frames):
            act = env.action_space.sample()
            
            obs_l.append(obs)
            act_l.append(act)

            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break

        total_frames += frame + 1
        print(f'rollout worker {worker_id} [{trial+1}/{trials}] ({total_frames} frames)')

        obs_l = np.array(obs_l, dtype=np.uint8)
        act_l = np.array(act_l, dtype=np.float16)
        np.savez_compressed(filename, obs=obs_l, act=act_l)

    env.close()
    return total_frames

def generate_rollouts(args):
    print("generating rollouts")
    if not os.path.exists(args.rollout_dir):
        os.mkdir(args.rollout_dir)

    workers = min(cpu_count(), args.n_workers)
    trials_per_worker = args.n_rollouts // workers + 1

    with Pool(workers) as pool:
        total_frames = sum(pool.starmap(rollout, [(i, trials_per_worker, args) for i in range(workers)]))
        print(f'Total frames collected at end: {total_frames}')

def train_vae(args):
    print("training vae")

    if not os.path.exists(args.dir_vae):
        os.mkdir(args.dir_vae)

    dataset = FrameDataset(f"{args.rollout_dir}/*.npz")
    print(f"compiled vae dataset ({len(dataset)} images)")

    workers = min(cpu_count(), args.n_workers)
    dataloader = DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True, num_workers=workers)
    print("created dataloader")

    config = VAEConfig()
    device = get_device(args)
    model = VAE(config).to(device)
    print(f"made vae ({sum(p.numel() for p in model.parameters())} parameters)")

    # Get all args that start with 'vae_' as a dict
    # Init wandb
    vae_args = {k.replace('vae_', ''): v for k, v in vars(args).items() if k.startswith('vae_')}
    wandb.init(project="world-models", config=vae_args, name=f"vae_{wandb.util.generate_id()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.vae_learning_rate)
    model.train()
    with tqdm(range(args.vae_epochs)) as t:
        for epoch in t:
            for x in dataloader:
                x = x.to(device)
                x = x.permute(0, 3, 1, 2).contiguous()  # Get channels to be first dimension
                x_hat = model(x)

                recon_loss = ((x - x_hat) ** 2).mean()
                kl_loss = args.vae_kl_weight * model.kl 
                loss = recon_loss + kl_loss
                opt.zero_grad()

                # Logging things
                t.set_postfix(
                    recon_loss=f"{recon_loss:6.4f}", kl_loss=f"{kl_loss:6.4f}"
                )
                wandb.log(
                    {
                        "recon_loss": recon_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "total_loss": loss.item(),
                    }
                )

                # Backprop
                loss.backward()
                opt.step()
            # Save each epoch's model
            torch.save(model.state_dict(), f"{args.dir_vae}/{epoch}.pth")
    wandb.finish()

def generate_latents(args):
    print("generating latents")
    return

def train_rnn(args):
    print("training rnn")
    return

def train_controller(args):
    print("training controller")
    return

# --------------------------------------------------------------------
# Main loop 

if __name__ == '__main__':
    import argparse

    # default settings will run all the phases in order
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument('--phases', nargs='+', choices=['rollout', 'vae', 'latent', 'rnn', 'controller'],
                    help='List of phases to run in order: rollout vae latent rnn controller')
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--env", type=str, default="racing", help="gym environment")
    # phase 1. extract rollouts
    parser.add_argument("--rollout-dir", type=str, default="rollouts", help="directory to save rollouts")
    parser.add_argument("--n-workers", type=int, default=64, help="number of workers to use for rollouts and loading")
    parser.add_argument("--n-rollouts", type=int, default=10_000, help="number of rollouts to extract")
    parser.add_argument("--max-frames", type=int, default=1000, help="maximum number of frames per rollout")
    # phase 2. train VAE
    parser.add_argument("--dir-vae", type=str, default="vae_checkpoints", help="directory to save VAE checkpoints")
    parser.add_argument("--vae-epochs", type=int, default=5, help="number of epochs to train the VAE")
    parser.add_argument("--vae-latent-dims", type=int, default=32, help="dimensionality of the latent space")
    parser.add_argument("--vae-batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--vae-kl-weight", type=float, default=0.00001, help="weight for KL divergence loss term")
    # phase 3. extract latent space dataset
    # TODO: Add arguments for this phase
    # phase 4. train RNN
    parser.add_argument("--rnn-dir", type=str, default="rnn_checkpoints", help="directory to save RNN checkpoints")
    parser.add_argument("--rnn-epochs", type=int, default=100, help="number of epochs to train the VAE")
    parser.add_argument("--rnn-hidden-size", type=int, default=128, help="dimensionality of the hidden state")

    # phase 5. train controller
    # TODO: Add arguments here
    
    args = parser.parse_args()

    # Do asserts
    print(f"using device: {get_device(args)}")
    assert args.env == "racing", "only racing is supported for now"
    # TODO: Actually use env for configs

    'rollout' in args.phases and generate_rollouts(args)
    'vae' in args.phases and train_vae(args)
    'latent' in args.phases and generate_latents(args)
    'rnn' in args.phases and train_rnn(args)
    'controller' in args.phases and train_controller(args)