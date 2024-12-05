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
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
    # else 'mps' if torch.backends.mps.is_available() else 'cpu'
)

wandb.init(project="world-models", config={
    "epochs": N_EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,  # Change to your actual batch size
    "latent_dims": LATENT_DIMS
})


def create_dataset(filelist, rollouts=10_000, steps=1_000):
    data = np.empty((rollouts*steps, 96, 96, 3), dtype=np.float32) 

    idx = 0
    for i in tqdm(range(rollouts)):
        filename = filelist[i]
        # obs = np.load(f'{DATA_DIR}/{filename}')['obs']
        obs = np.load(os.path.join(DATA_DIR, filename))['obs']
        # AHH DONT FORGET TO NORMALIZE
        obs = obs.astype(np.float32) / 255.0
        obs_len = len(obs)
        if idx + obs_len > rollouts * steps:
            data = data[:idx]
            print('Exiting with too much data')
            break
        data[idx:idx+obs_len] = obs
        idx += obs_len

    return ObsDataset(data)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    with tqdm(range(epochs)) as t:
        for epoch in t:
            for b_idx, x in enumerate(data):
                x = x.to(device)
                x = x.permute(0, 3, 1, 2)  # Get channels to be first dimension
                opt.zero_grad()
                x_hat = autoencoder(x)

                recon_loss = ((x - x_hat) ** 2).mean() 
                # tuning this beta so they're roughly equal losses to begin with?
                kl_loss = 0.005 * autoencoder.encoder.kl / x.shape[0]
                loss = recon_loss + kl_loss
                # print(((x - x_hat) ** 2).sum(), autoencoder.encoder.kl)
                # print(f'epoch {epoch} | loss {loss:.2f}')
                t.set_postfix(recon_loss=f'{recon_loss:6.4f}', kl_loss=f'{kl_loss:6.4f}')
                wandb.log({
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "total_loss": loss.item(),
                })
                # if b_idx % 10 == 0:
                #     plt.imshow(x[0].permute(1, 2, 0).cpu().detach().numpy())
                #     plt.show()
                loss.backward()
                opt.step()
            torch.save(autoencoder.state_dict(), f'{CHECKPOINT_DIR}/{epoch}.pth')
    wandb.finish()
    return autoencoder


def main():
    file_list = os.listdir(DATA_DIR)
    dataset = create_dataset(file_list, rollouts=len(file_list))
    print(f'> Compiled datsaet! {len(dataset)} images')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('> Created dataloader!')
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # data_iter = iter(dataloader)
    # images_batch = next(data_iter)

    # print(images_batch.shape)
    # # Plot the batch of images
    # fig, axes = plt.subplots(1, len(images_batch), figsize=(15, 5))

    # for i, ax in enumerate(axes):
    #     im = images_batch[i].numpy()
    #     print(np.ptp(im))
    #     print(np.min(im))
    #     print(np.max(im))
    #     print(not np.any(im))
    #     ax.imshow(
    #         # images_batch[i].permute(1, 2, 0).numpy()
    #         im
    #     )  # Convert from (C, H, W) to (H, W, C)
    #     ax.axis("off")

    # plt.show()

    model = VAE(latent_dims=LATENT_DIMS, in_c=3).to(device)
    print(f'> Made model! ({sum(p.numel() for p in model.parameters())} parameters)')
    train(model, dataloader, N_EPOCHS)


if __name__ == '__main__':
    main()
