import numpy as np
import os
import torch

import matplotlib.pyplot as plt
from models.vae import VAE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb


class ObsDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


LATENT_DIMS = 32
BATCH_SIZE = 128
LR = 1e-4
N_EPOCHS = 10
DATA_DIR = 'rollouts'
CHECKPOINT_DIR = 'vae_checkpoints'

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
        "batch_size": BATCH_SIZE,  # Change to your actual batch size
        "latent_dims": LATENT_DIMS,
    },
)


def create_dataset(filelist, rollouts=10_000, steps=1_000):
    data = np.empty((rollouts * steps, 96, 96, 3), dtype=np.float32)

    idx = 0
    for i in tqdm(range(rollouts)):
        filename = filelist[i]
        obs = np.load(os.path.join(DATA_DIR, filename))["obs"]
        # AHH DONT FORGET TO NORMALIZE
        obs = obs.astype(np.float32) / 255.0
        obs_len = len(obs)
        if idx + obs_len > rollouts * steps:
            data = data[:idx]
            print("Exiting with too much data")
            break
        data[idx : idx + obs_len] = obs
        idx += obs_len

    return ObsDataset(data)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    with tqdm(range(epochs)) as t:
        for epoch in t:
            for _, x in enumerate(data):
                x = x.to(device)
                x = x.permute(0, 3, 1, 2)  # Get channels to be first dimension
                opt.zero_grad()
                x_hat = autoencoder(x)

                recon_loss = ((x - x_hat) ** 2).mean()
                # tuning this beta so they're roughly equal losses to begin with?
                kl_weight = 0.00001
                kl_loss = kl_weight * autoencoder.kl
                loss = recon_loss + kl_loss
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
                loss.backward()
                opt.step()
            torch.save(autoencoder.state_dict(), f"{CHECKPOINT_DIR}/{epoch}.pth")
    wandb.finish()
    return autoencoder


def main():
    file_list = os.listdir(DATA_DIR)
    files_taken = 200
    dataset = create_dataset(file_list[:files_taken], rollouts=files_taken)
    print(f"> Compiled datsaet! {len(dataset)} images")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("> Created dataloader!")

    model = VAE(latent_dims=LATENT_DIMS, in_c=3).to(device)
    print(f"> Made model! ({sum(p.numel() for p in model.parameters())} parameters)")
    train(model, dataloader, N_EPOCHS)


if __name__ == "__main__":
    main()
