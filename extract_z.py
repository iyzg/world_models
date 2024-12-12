# Get VAE mu and logvar out of the dataset
import numpy as np
import os
import random

from models.vae import VAE
from tqdm import trange
import torch
torch.set_grad_enabled(False)

DATA_DIR = "rollouts"
CHECKPOINTS_DIR = "vae_checkpoints"
Z_DATA_DIR = "series"
LATENT_DIMS = 32
MODEL_CHECKPOINT_NUM = 6
FILES_TO_USE = 200
BATCH_SIZE = 1_000


# Load dataset
def load_data(filelist, steps=1_000):
    rollouts = min(FILES_TO_USE, len(filelist))
    obs_l = np.empty((rollouts * steps, 96, 96, 3), dtype=np.float32)
    act_l = np.empty((rollouts * steps, 3), dtype=np.float32)

    o_idx = 0
    a_idx = 0
    for i in trange(rollouts):
        filename = filelist[i]
        data = np.load(f"{DATA_DIR}/{filename}")
        obs = data["obs"]
        act = data["act"]

        # AHH DONT FORGET TO NORMALIZE
        obs = obs.astype(np.float32) / 255.0
        obs_len = len(obs)
        act_len = len(act)
        if o_idx + obs_len > rollouts * steps or a_idx + act_len > rollouts * steps:
            print("Exiting with too much data")
            exit(0)

        obs_l[o_idx : o_idx + obs_len] = obs
        act_l[a_idx : a_idx + act_len] = act
        o_idx += obs_len
        a_idx += act_len

    return obs_l, act_l


# Load model
def load_vae():
    model = VAE(latent_dims=LATENT_DIMS, in_c=3)
    model.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINTS_DIR, f"{MODEL_CHECKPOINT_NUM}.pth"),
            weights_only=True,
        )
    )
    model.eval()
    return model


# Save the new one where array is [action, mu, logvar]
def main():
    filelist = os.listdir(DATA_DIR)
    obs_l, act_l = load_data(filelist)
    model = load_vae()

    mu_l = np.empty((len(obs_l), 32), dtype=np.float32)
    logvar_l = np.empty((len(obs_l), 32), dtype=np.float32)
    for b in trange(len(obs_l) // BATCH_SIZE + 1):
        lb, ub = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        obs = obs_l[lb:ub]
        mu, logvar = model.encode(
            torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)
        )
        mu_l[lb:ub] = mu.detach().numpy()
        logvar_l[lb:ub] = logvar.detach().numpy()

    if not os.path.exists(Z_DATA_DIR):
        os.mkdir(Z_DATA_DIR)

    np.savez_compressed(
        os.path.join(Z_DATA_DIR, "series.npz"), act=act_l, mu=mu_l, logvar=logvar_l
    )


if __name__ == "__main__":
    main()
