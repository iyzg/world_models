# Train VAE off random rollouts
from multiprocessing import Pool, cpu_count

import gymnasium as gym
import numpy as np
import os
import random
import torch
torch.set_grad_enabled(False)

from models.rnn import MDNRNN
from models.vae import VAE
from models.controller import Controller

# Constants
# TODO: Turn this into command line prompts
# TODO: Also save action rather than just the thing, since right now I just want to
# get the datset to train
DIR_NAME = "rollouts"
NUM_WORKERS = min(64, cpu_count())

MAX_FRAMES = 1_000
TOTAL_ROLLOUTS = 9_500
TRIALS_PER_WORKER = TOTAL_ROLLOUTS // NUM_WORKERS + 1

USE_AGENT = False

#### USE AGENT PARAMETERS ####

VAE_CHECKPOINTS = 'vae_checkpoints'
RNN_CHECKPOINTS = 'rnn_checkpoints'
AGENT_CHECKPOINTS = 'agent_checkpoints'

VAE_CHECKPOINT_N = '9'
RNN_CHECKPOINT_N = '450'

# TODO: load the VAE and RNN and env
VAE_LATENT_DIMS = 32 


RNN_INPUT_SIZE = 32 + 3 # latent_dims + action_size
RNN_HIDDEN_SIZE = 128
RNN_OUTPUT_SIZE = 32
N_GAUSSIANS = 5
N_LAYERS = 1


POP_SIZE = 32
ACT_INPUT_SIZE = RNN_HIDDEN_SIZE + VAE_LATENT_DIMS
ACT_SPACE = 3

####

if not os.path.exists(DIR_NAME):
    os.mkdir(DIR_NAME)


def generate_rollouts(worker_id):
    if USE_AGENT:
        vae = VAE(latent_dims=32, in_c=3)
        vae.load_state_dict(torch.load(os.path.join(VAE_CHECKPOINTS, f'{VAE_CHECKPOINT_N}.pth'), weights_only=True))
        vae.eval()

        rnn = MDNRNN(N_GAUSSIANS, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_OUTPUT_SIZE, N_LAYERS)
        rnn.load_state_dict(torch.load(os.path.join(RNN_CHECKPOINTS, f'{RNN_CHECKPOINT_N}.pth'), weights_only=True))
        rnn.eval()

        c_data = np.load(os.path.join(AGENT_CHECKPOINTS, 'best_controller.npy'))
        controller = Controller(c_data, ACT_INPUT_SIZE, ACT_SPACE)

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    TOTAL_FRAMES = 0

    for trial in range(TRIALS_PER_WORKER):
        seed = random.randint(0, 2**31 - 1)
        filename = f'{DIR_NAME}/{worker_id}_{seed}.npz'
        obs_l = []
        act_l = []
        if USE_AGENT:
            h = rnn.initial_state(1)

        # NOTE: obs is 96x96
        obs, info = env.reset(seed=seed)
        for frame in range(MAX_FRAMES):
            if not USE_AGENT:
                act = env.action_space.sample()
            else:
                mu, logvar = vae.encode(torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))
                z = vae.reparamterize(mu, logvar)
                c_in = torch.cat([z.flatten(), h[1].flatten()], dim=-1).numpy()
                act = controller.forward(c_in)

            obs_l.append(obs)
            act_l.append(act)

            obs, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break

            if USE_AGENT:
                _, _, _, h = rnn(torch.cat([z.flatten(), torch.tensor(act, dtype=torch.float32)], dim=-1).unsqueeze(0).unsqueeze(0), h)

        TOTAL_FRAMES += frame + 1
        print(f'Worker {worker_id} [{trial+1}/{TRIALS_PER_WORKER}] ({TOTAL_FRAMES} frames)')

        obs_l = np.array(obs_l, dtype=np.uint8)
        act_l = np.array(act_l, dtype=np.float16)
        np.savez_compressed(filename, obs=obs_l, act=act_l)

    env.close()
    return TOTAL_FRAMES


def main():
    with Pool(NUM_WORKERS) as pool:
        total_frames = sum(pool.map(generate_rollouts, range(NUM_WORKERS)))
        print(f'Total frames collected at end: {total_frames}')


if __name__ == '__main__':
    main()