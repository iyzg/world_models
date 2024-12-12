# TODO: Agent training
import cma
import numpy as np
from models.rnn import MDNRNN
from models.vae import VAE
from models.controller import Controller
import torch
torch.set_grad_enabled(False)
import os
import gymnasium as gym

import multiprocessing as mp
from multiprocessing import Pool, cpu_count

VAE_CHECKPOINTS = 'vae_checkpoints'
RNN_CHECKPOINTS = 'rnn_checkpoints'
AGENT_CHECKPOINTS = 'agent_checkpoints'

VAE_CHECKPOINT_N = '9'
RNN_CHECKPOINT_N = '450'

# TODO: load the VAE and RNN and env
VAE_LATENT_DIMS = 32 


RNN_INPUT_SIZE = 32 + 3 # latent_dims + action_size
RNN_HIDDEN_SIZE = 64
RNN_OUTPUT_SIZE = 32
N_GAUSSIANS = 5
N_LAYERS = 1


POP_SIZE = 32
ACT_INPUT_SIZE = RNN_HIDDEN_SIZE + VAE_LATENT_DIMS
ACT_SPACE = 3

def sigmoid(x):
    return np.clip(0.5 * (1 + np.tanh(x / 2)), 0, 1)

def get_action(c, latents):
    w = c[:ACT_INPUT_SIZE * ACT_SPACE].reshape(ACT_INPUT_SIZE, ACT_SPACE)
    b = c[ACT_INPUT_SIZE * ACT_SPACE:]
    out = np.dot(latents, w) + b
    out = np.nan_to_num(out, nan=0.0)
    out[0] = np.tanh(out[0])
    out[1] = sigmoid(out[1])
    out[2] = sigmoid(out[2])
    return out


def rollout(controller, n_trials = 8):
    vae = VAE(latent_dims=VAE_LATENT_DIMS, in_c=3)
    vae.load_state_dict(torch.load(os.path.join(VAE_CHECKPOINTS, f'{VAE_CHECKPOINT_N}.pth'), weights_only=True))
    vae.eval()

    rnn = MDNRNN(N_GAUSSIANS, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_OUTPUT_SIZE, N_LAYERS)
    rnn.load_state_dict(torch.load(os.path.join(RNN_CHECKPOINTS, f'{RNN_CHECKPOINT_N}.pth'), weights_only=True))
    rnn.eval()

    rewards = []
    for trial in range(n_trials):
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        )

        obs, info = env.reset()
        h = rnn.initial_state(1)
        done = False
        cumulative_reward = 0
        while not done:
            frame = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            mu, logvar = vae.encode(frame)
            z = vae.reparamterize(mu, logvar)
            c_in = torch.cat([z.flatten(), h[1].flatten()], dim=-1).numpy()
            a = get_action(controller, c_in)
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
            cumulative_reward += reward
            _, _, _, h = rnn(torch.cat([z.flatten(), torch.tensor(a, dtype=torch.float32)], dim=-1).unsqueeze(0).unsqueeze(0), h)
        rewards.append(cumulative_reward)
        env.close()
    # print('Evaled agent with reward:', cumulative_reward)
    avg_reward = sum(rewards) / len(rewards)
    return -avg_reward  # Negative since CMA minimizes


def parallel_rollout(controllers, num_processes=None):
    """
    Evaluate multiple controllers in parallel using multiprocessing
    
    Args:
        controllers: List of controller parameters to evaluate
        num_processes: Number of processes to use (defaults to CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()  # Use all available CPU cores
    
    # Create a process pool
    with mp.Pool(processes=num_processes) as pool:
        # Map controllers to rollout function
        rewards = pool.map(rollout, controllers)
    
    return rewards


def main():
    if not os.path.exists(AGENT_CHECKPOINTS):
        os.mkdir(AGENT_CHECKPOINTS)

    best_reward = float('-inf')
    best_solution = None
    # es = cma.CMAEvolutionStrategy((ACT_INPUT_SIZE + 1) * ACT_SPACE * [0], 0.5, {'popsize': 64})
    es = cma.CMAEvolutionStrategy((ACT_INPUT_SIZE + 1) * ACT_SPACE * [0], 0.5)
    while not es.stop():
        solutions = es.ask()
        rewards = parallel_rollout(solutions)

        # Find best solution in this generation
        min_idx = np.argmin(rewards)  # Using min since rewards are negative
        gen_best_reward = -rewards[min_idx]  # Convert back to positive
        gen_best_solution = solutions[min_idx]
        
        # Update overall best if we found a better solution
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_solution = gen_best_solution
            # Save the best solution to disk
            np.save(f'{AGENT_CHECKPOINTS}/best_controller.npy', best_solution)
            print('New best solution found with reward:', best_reward)
            
        es.tell(solutions, rewards)
        es.logger.add()
        es.disp()
    es.result_pretty()
    es.logger.plot()
  
if __name__ == '__main__':
    main()