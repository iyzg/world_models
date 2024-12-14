# Normal comments
## Teaching comments

import bisect
import glob
import math
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

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

# TODO: Incorporate temp into the MDN model
# TODO: Clean up all this code
class MDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.output_size
        self.output_size = config.output_size
        self.n_gaussians = config.n_gaussians 

        self.pi = nn.Linear(self.input_size, self.n_gaussians)
        self.mu = nn.Linear(self.input_size, self.n_gaussians * self.output_size)
        self.sigma_lin = nn.Linear(self.input_size, self.n_gaussians * self.output_size)

        self.temp = config.temp 

    def forward(self, x):
        # Drop down to one timestep at a time
        x = x.squeeze(0)
        pi = F.softmax(self.pi(x), dim=1)

        mu = self.mu(x)
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
    def __init__(self, config):
        super().__init__()
        self.layers = config.n_layers
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size 
        self.rnn = nn.LSTM(config.input_size, config.hidden_size, config.n_layers, batch_first=True, proj_size=config.output_size)
        self.mdn = MDN(config)

    def forward(self, x, h):
        y, h = self.rnn(x, h)
        pi, mu, sigma = self.mdn(y)
        return pi, mu, sigma, h

    def initial_state(self, batch_size):
        return (torch.zeros(self.layers, batch_size, self.output_size),
                torch.zeros(self.layers, batch_size, self.hidden_size))

# --------------------------------------------------------------------
# Controller

# Implement controller config class
@dataclass
class ControllerConfig:
    input_size: int = 32 + 3
    act_size: int = 3

class Controller():
    def __init__(self, data, config):
        self.input_size = config.input_size
        self.act_size = config.act_size
        self.w = data[:self.input_size * self.act_size].reshape(self.input_size, self.act_size)
        self.b = data[self.input_size * self.act_size:]

    def __call__(self, x):
        x = np.dot(x, self.w) + self.b
        x = np.nan_to_num(x, nan=0.0)

        # Ensure it gives right output range
        x[0] = np.tanh(x[0])
        x[1] = self._sigmoid(x[1])
        x[2] = self._sigmoid(x[2])

        return x 

    def _sigmoid(self, x):
        return np.clip(0.5 * (1 + np.tanh(x / 2)), 0, 1)

# --------------------------------------------------------------------
# Dataset classes

def count_frames(filename):
    return len(np.load(filename)["obs"])

class FrameDataset(Dataset):
    def __init__(self, file_pattern):
        # TODO: Add preprocessed path to args
        self.preprocessed_path = Path("rollouts.npy")
        if self.preprocessed_path.exists():
            print(f"Loading preprocessed data from {self.preprocessed_path}")
            self.data = np.load(self.preprocessed_path)
            self.total_frames = len(self.data)
            self.data = torch.from_numpy(self.data)
            return

        files = sorted(glob.glob(file_pattern))

        with Pool() as pool:
            frame_counts = list(pool.map(count_frames, files))

        self.total_frames = sum(frame_counts)
        
        # Pre-allocate array with proper type
        sample = np.load(files[0])["obs"]
        self.data = np.empty((self.total_frames, *sample.shape[1:]), dtype=np.float32)
            
        current_idx = 0
        for file in files:
            frames = np.load(file)["obs"].astype(np.float32) / 255.0
            length = len(frames)
            self.data[current_idx:current_idx + length] = frames
            current_idx += length

        # Save the preprocessed data
        print(f"Saving preprocessed data to {self.preprocessed_path}")
        np.save(self.preprocessed_path, self.data)
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return self.total_frames
        
    def __getitem__(self, idx):
        return self.data[idx]

class LatentDatset(Dataset):
    def __init__(self, file_pattern):
        self.file_list = sorted(glob.glob(file_pattern))
        self.n_episodes = len(self.file_list)

        # Load one file to get dimensions
        sample_data = np.load(self.file_list[0])
        latent_dim = sample_data["mu"].shape[1]
        action_dim = sample_data["act"].shape[1]
        seq_length = sample_data["mu"].shape[0]
        n_episodes = len(self.file_list)

        # Pre-allocate tensors
        self.mus = torch.empty(n_episodes, seq_length, latent_dim)
        self.logvars = torch.empty(n_episodes, seq_length, latent_dim)
        self.actions = torch.empty(n_episodes, seq_length, action_dim)
        
        # Load all data
        for idx, file in enumerate(tqdm(self.file_list)):
            data = np.load(file)
            self.mus[idx] = torch.from_numpy(data["mu"])
            self.logvars[idx] = torch.from_numpy(data["logvar"])
            self.actions[idx] = torch.from_numpy(data["act"])
        
        self.stds = torch.exp(0.5 * self.logvars)

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        # Sample from the latent distribution during access
        z = self.mus[idx] + self.stds[idx] * torch.randn_like(self.stds[idx])
        
        inputs = torch.cat((z[:-1], self.actions[idx][:-1]), dim=-1)
        
        return inputs, z[1:]

# ---------------------------------------------------------------------
# Environment dataclass

@dataclass
class EnvConfig:
    input_size: tuple = (96, 96)
    in_channels: int = 3
    act_size: int = 3

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
@torch.no_grad()
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

@torch.no_grad()
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
    if args.compile:
        model = torch.compile(model)

    # Get all args that start with 'vae_' as a dict
    # Init wandb
    vae_args = {k.replace('vae_', ''): v for k, v in vars(args).items() if k.startswith('vae_')}
    wandb.init(project="world-models", config=vae_args, name=f"vae_{wandb.util.generate_id()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.vae_learning_rate)
    model.train()
    with tqdm(range(args.vae_epochs + 1)) as t:
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

@torch.no_grad()
def generate_latents(args):
    print("generating latents")
    # TODO: load vae more generally
    # the config should adapt to the args somehow
    config = VAEConfig()
    device = get_device(args)
    model = VAE(config).to(device)

    if args.compile:
        model = torch.compile(model)
    # load the most recent checkpoint (assuming numbered correctly)
    checkpoints = sorted(glob.glob(f"{args.dir_vae}/*.pth"))
    model.load_state_dict(
        torch.load(
            checkpoints[-1],
            weights_only=True,
        )
    )
    model.eval()

    if not os.path.exists(args.dir_latent):
        os.mkdir(args.dir_latent)

    # loop through rollouts, get obs, get latent, save
    for file in glob.glob(f"{args.rollout_dir}/*.npz"):
        data = np.load(file)
        obs = data["obs"]
        act = data["act"]

        # AHH DONT FORGET TO NORMALIZE
        obs = obs.astype(np.float32) / 255.0
        mu, logvar = model.encode(
            torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        )

        # Save to .npz file
        latent_file = f"{args.dir_latent}/{file.split('/')[-1]}"
        np.savez_compressed(latent_file, mu=mu.cpu().numpy(), logvar=logvar.cpu().numpy(), act=act)

def train_rnn(args):
    print("training rnn")
    if not os.path.exists(args.dir_rnn):
        os.mkdir(args.dir_rnn)

    dataset = LatentDatset(f"{args.dir_latent}/*.npz")
    print(f"compiled rnn dataset ({len(dataset)} episodes)")

    workers = min(cpu_count(), args.n_workers)
    dataloader = DataLoader(dataset, batch_size=args.rnn_batch_size, shuffle=True, num_workers=workers)
    print("created dataloader")

    # TODO: Figure out way to actually use config
    config = RNNConfig()
    device = get_device(args)
    if device == "mps":
        print("Warning: MPS is not supported for LSTM training, defaulting to CPU")
        device = "cpu"
    model = MDNRNN(config).to(device)
    if args.compile:
        model = torch.compile(model)

    rnn_args = {k.replace('rnn_', ''): v for k, v in vars(args).items() if k.startswith('rnn_')}
    wandb.init(project="world-models", config=rnn_args, name=f"rnn_{wandb.util.generate_id()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.rnn_learning_rate)
    model.train()
    with tqdm(range(args.rnn_epochs + 1)) as t:
        for epoch in t:
            for (x, y) in dataloader:
                x, y = x.to(device), y.to(device)
                h = model.initial_state(x.size(0))

                opt.zero_grad()
                loss = 0.0
                for ts in range(x.size(1)):
                    pi, mu, sigma, h = model(x[:, ts].unsqueeze(1), h)
                    step_loss = model.mdn.loss(pi, mu, sigma, y[:, ts])
                    loss += step_loss

                # Average loss over timesteps
                loss /= x.size(1)

                t.set_postfix(
                    loss = f"{loss:6.4f}",
                )
                wandb.log(
                    {
                        "loss": loss.item()
                    }
                )
                loss.backward()
                opt.step()

            # TODO: Add command for how often to save model
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"{args.dir_rnn}/{epoch}.pth")
    wandb.finish()

@torch.no_grad()
def rollout(solution, args):
    device = get_device(args)
    if device == "mps":
        print("Warning: MPS is not supported for LSTM training, defaulting to CPU")
        device = "cpu"

    vae = VAE(VAEConfig()).to(device)
    rnn = MDNRNN(RNNConfig()).to(device)
    c_config = ControllerConfig(input_size = args.rnn_hidden_size + args.vae_latent_dims)
    controller = Controller(solution, c_config)

    if args.compile:
        vae = torch.compile(vae)
        rnn = torch.compile(rnn)

    # load the most recent checkpoint (assuming numbered correctly)
    # TODO: Rewrite this into the class
    checkpoints = sorted(glob.glob(f"{args.dir_vae}/*.pth"))
    vae.load_state_dict(
        torch.load(
            checkpoints[-1],
            weights_only=True,
        )
    )
    vae.eval()
    checkpoints = sorted(glob.glob(f"{args.dir_rnn}/*.pth"))
    rnn.load_state_dict(
        torch.load(
            checkpoints[-1],
            weights_only=True,
        )
    )
    rnn.eval()

    rewards = []
    for _ in range(args.controller_evals_per_agent):
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        )

        obs, _ = env.reset()
        h = rnn.initial_state(1)
        h = (h[0].to(device), h[1].to(device))
        done = False
        cumulative_reward = 0
        while not done:
            frame = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            frame = frame.to(device)
            mu, logvar = vae.encode(frame)
            z = vae.reparamterize(mu, logvar)
            c_in = torch.cat([z.flatten(), h[1].flatten()], dim=-1).cpu().numpy()
            a = controller(c_in)
            obs, reward, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
            cumulative_reward += reward
            a = torch.tensor(a, dtype=torch.float32, device=device)
            _, _, _, h = rnn(torch.cat([z.flatten(), a], dim=-1).unsqueeze(0).unsqueeze(0), h)
            # h = (h[0].to(device), h[1].to(device))
        rewards.append(cumulative_reward)
        env.close()
    print('Evaled agent with reward:', cumulative_reward)
    avg_reward = sum(rewards) / len(rewards)
    return -avg_reward  # Negative since CMA minimizes

@torch.no_grad()
def train_controller(args):
    print("training controller")
    if not os.path.exists(args.dir_controller):
        os.mkdir(args.dir_controller)

    best_reward = float('-inf')
    best_solution = None

    workers = min(cpu_count(), args.n_workers)
    es = cma.CMAEvolutionStrategy((c_config.input_size * c_config.act_size + c_config.act_size) * [0], 0.5)
    while not es.stop():
        solutions = es.ask()
        with Pool(workers) as pool:
            rewards = pool.starmap(rollout, [(s, args) for s in solutions])

        # Find best solution in this generation
        min_idx = np.argmin(rewards)  # Using min since rewards are negative
        gen_best_reward = -rewards[min_idx]  # Convert back to positive
        gen_best_solution = solutions[min_idx]
        
        # Update overall best if we found a better solution
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_solution = gen_best_solution
            # Save the best solution to disk
            np.save(f'{args.dir_controller}/best_controller.npy', best_solution)
            print('New best solution found with reward:', best_reward)
            
        es.tell(solutions, rewards)
        es.logger.add()
        es.disp()

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
    parser.add_argument("--n-rollouts", type=int, default=5_000, help="number of rollouts to extract")
    parser.add_argument("--max-frames", type=int, default=1000, help="maximum number of frames per rollout")
    # phase 2. train VAE
    # TODO: add thing for kernel sizes and stride
    parser.add_argument("--dir-vae", type=str, default="vae_checkpoints", help="directory to save VAE checkpoints")
    parser.add_argument("--vae-latent-dims", type=int, default=32, help="dimensionality of the latent space")

    parser.add_argument("--vae-epochs", type=int, default=5, help="number of epochs to train the VAE")
    parser.add_argument("--vae-batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--vae-learning-rate", type=float, default=0.0001, help="learning rate for training")
    parser.add_argument("--vae-kl-weight", type=float, default=0.00001, help="weight for KL divergence loss term")
    # phase 3. extract latent space dataset
    parser.add_argument("--dir-latent", type=str, default="series", help="directory to save latent space dataset")
    # phase 4. train RNN
    parser.add_argument("--dir-rnn", type=str, default="rnn_checkpoints", help="directory to save RNN checkpoints")
    parser.add_argument("--rnn-hidden-size", type=int, default=128, help="dimensionality of the hidden state")
    parser.add_argument("--rnn-n-gaussians", type=int, default=5, help="number of gaussians for the MDN")
    parser.add_argument("--rnn-layers", type=int, default=1, help="number of layers in the RNN")
    parser.add_argument("--rnn-temp", type=float, default=1.0, help="temperature for the MDN")

    parser.add_argument("--rnn-epochs", type=int, default=100, help="number of epochs to train the VAE")
    parser.add_argument("--rnn-batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--rnn-learning-rate", type=float, default=1e-3, help="learning rate for training")
    # phase 5. train controller
    parser.add_argument("--dir-controller", type=str, default="controller_checkpoints", help="directory to save controller checkpoints")
    parser.add_argument("--controller-pop-size", type=int, default=64, help="population size for controller training")
    parser.add_argument("--controller-evals-per-agent", type=int, default=8, help="number of evaluations per agent")
    
    args = parser.parse_args()

    # Do asserts
    print(f"using device: {get_device(args)}")
    assert args.env == "racing", "only racing is supported for now"

    # TODO: Actually use env for configs
    # IDEA: Create config here and then pass it into functions that need it

    'rollout' in args.phases and generate_rollouts(args)
    'vae' in args.phases and train_vae(args)
    'latent' in args.phases and generate_latents(args)
    'rnn' in args.phases and train_rnn(args)
    'controller' in args.phases and train_controller(args)