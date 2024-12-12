# TODO: Train MDN-RNN

import numpy as np
import os
import torch

from models.rnn import MDNRNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb


class LatentSeriesDataset(Dataset):
    def __init__(self, mu, logvar, act, n_episodes, episode_length):
        self.mu = torch.tensor(mu.reshape(n_episodes, episode_length, -1))
        self.logvar = torch.tensor(logvar.reshape(n_episodes, episode_length, -1))
        self.act = torch.tensor(act.reshape(n_episodes, episode_length, -1))

        self.n_episodes = n_episodes
        self.episode_length = episode_length


    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        episode_mu = self.mu[idx]
        episode_logvar = self.logvar[idx]
        episode_act = self.act[idx]

        # Sample from the latent
        std = torch.exp(0.5 * episode_logvar)
        eps = torch.randn_like(episode_mu)
        z = eps * std + episode_mu

        # Concatenate the latent with the action
        inputs = torch.cat((z[:-1], episode_act[:-1]), dim=-1)
        targets = z[1:]
        return inputs, targets

# MDN-RNN parameters
RNN_INPUT_SIZE = 32 + 3 # latent_dims + action_size
RNN_HIDDEN_SIZE = 64
RNN_OUTPUT_SIZE = 32
N_GAUSSIANS = 5
N_LAYERS = 1

N_EPISODES = 200
EPISODE_LENGTH = 1000
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 500
DATA_DIR = 'series'
CHECKPOINT_DIR = 'rnn_checkpoints'

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    # else 'mps' if torch.backends.mps.is_available() else 'cpu'
)

wandb.init(
    project="world-models",
    config={
        "epochs": N_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "hidden_size": RNN_HIDDEN_SIZE,
        "n_layers": N_LAYERS,
        "n_gaussians": N_GAUSSIANS,
    },
    name=f"RNN_{wandb.util.generate_id()}"
)

def train(model, data, epochs=20):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    with tqdm(range(epochs + 1)) as t:
        for epoch in t:
            for _, (x, y) in enumerate(data):
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
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{epoch}.pth")
    wandb.finish()
    return model 


def main():
    raw_data = np.load(os.path.join(DATA_DIR, 'series.npz'))
    data_mu = raw_data['mu']
    data_logvar = raw_data['logvar']
    data_act = raw_data['act']
    # print(data_mu.shape, data_logvar.shape, data_act.shape)
    dataset = LatentSeriesDataset(data_mu, data_logvar, data_act, N_EPISODES, EPISODE_LENGTH)
    print("> Created dataset!")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("> Created dataloader!")

    model = MDNRNN(N_GAUSSIANS, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_OUTPUT_SIZE, N_LAYERS).to(device)
    print(f"> Made model! ({sum(p.numel() for p in model.parameters())} parameters)")
    train(model, dataloader, N_EPOCHS)


if __name__ == "__main__":
    main()
