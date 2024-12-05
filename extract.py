# Train VAE off random rollouts
from multiprocessing import Pool, cpu_count

import gymnasium as gym
import numpy as np
import os
import random

# Constants
# TODO: Turn this into command line prompts
# TODO: Also save action rather than just the thing, since right now I just want to
# get the datset to train
DIR_NAME = "rollouts"
NUM_WORKERS = min(64, cpu_count())

MAX_FRAMES = 1_000
TOTAL_ROLLOUTS = 500
TRIALS_PER_WORKER = TOTAL_ROLLOUTS // NUM_WORKERS + 1

if not os.path.exists(DIR_NAME):
    os.mkdir(DIR_NAME)


def generate_rollouts(worker_id):
    env = gym.make(
        "CarRacing-v2",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False,
    )

    TOTAL_FRAMES = 0

    for trial in range(TRIALS_PER_WORKER):
        seed = random.randint(0, 2**31 - 1)
        filename = f'{DIR_NAME}/{worker_id}_{seed}.npz'
        obs_l = []
        act_l = []

        # NOTE: obs is 96x96
        obs, info = env.reset(seed=seed)
        for frame in range(MAX_FRAMES):
            act = env.action_space.sample()

            obs_l.append(obs)
            act_l.append(act)

            obs, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break

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