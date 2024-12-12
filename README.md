# World Models

A personal implementation of David Ha's [world model](https://worldmodels.github.io/) architecture!

When I clean up the code, I'll add command line options. Right now, rollouts should go to `/rollouts` and VAE checkpoints go to `/vae_checkpoints`.

## Order of commands

`python extract.py`

`python vae_train.py`

`python extract_z.py`

`python rnn_train.py`

`python agent_train.py`
