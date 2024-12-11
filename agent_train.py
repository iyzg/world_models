# TODO: Agent training
import cma
import numpy as np
from models.rnn import MDNRNN
from models.vae import VAE
import torch
torch.set_grad_enabled(False)
import os
import gymnasium as gym

VAE_CHECKPOINTS = 'vae_checkpoints'
RNN_CHECKPOINTS = 'rnn_checkpoints'

VAE_CHECKPOINT_N = '9'
RNN_CHECKPOINT_N = '50'

# TODO: load the VAE and RNN and env
VAE_LATENT_DIMS = 32 
vae = VAE(latent_dims=VAE_LATENT_DIMS, in_c=3)
vae.load_state_dict(torch.load(os.path.join(VAE_CHECKPOINTS, f'{VAE_CHECKPOINT_N}.pth'), weights_only=True))
vae.eval()

RNN_INPUT_SIZE = 32 + 3 # latent_dims + action_size
RNN_HIDDEN_SIZE = 128
RNN_OUTPUT_SIZE = 32
N_GAUSSIANS = 5
N_LAYERS = 1

rnn = MDNRNN(N_GAUSSIANS, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_OUTPUT_SIZE, N_LAYERS)
rnn.load_state_dict(torch.load(os.path.join(RNN_CHECKPOINTS, f'{RNN_CHECKPOINT_N}.pth'), weights_only=True))
rnn.eval()

# TODO
env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

POP_SIZE = 32
ACT_INPUT_SIZE = RNN_HIDDEN_SIZE + VAE_LATENT_DIMS
ACT_SPACE = 3

def get_action(c, latents):
    w = c[:ACT_INPUT_SIZE * ACT_SPACE].reshape(ACT_INPUT_SIZE, ACT_SPACE)
    b = c[ACT_INPUT_SIZE * ACT_SPACE:]
    out = np.dot(latents, w) + b
    out = np.nan_to_num(out, nan=0.0)
    out[0] = np.tanh(out[0])
    out[1] = (out[1] + 1.0) / 2.0
    out[2] = (out[2] + 1.0) / 2.0
    return out

def rollout(controller):
    ''' env, rnn, vae are '''
    ''' global variables  '''
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
        print(cumulative_reward)
    return cumulative_reward

def main():
    es = cma.CMAEvolutionStrategy((ACT_INPUT_SIZE + 1) * ACT_SPACE * [0], 0.5)
    while not es.stop():
        solutions = es.ask()
        rewards = [rollout(s) for s in solutions]
        es.tell(solutions, rewards)
        es.logger.add()
        es.disp()
    es.result_pretty()
    es.logger.plot()
  
if __name__ == '__main__':
    main()