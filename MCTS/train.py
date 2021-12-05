# %%
from games.tic_tac_toe import TicTacToe
from networks.tic_tac_toe_net import create_network

from model import Model
import numpy as np

np.random.seed(42)

# %%
game = TicTacToe()
model = Model(create_network(game.shape, game.nb_actions), summary=True)

# %%
accept_model_params = {"nb_games": 20, "nb_simulations": 25, "min_win_ratio": 0.51}
learn_params = {
  "nb_iter": 30,
  "nb_simulations": 25,
  "nb_games": 100,
  "nb_epochs": 20,
  "maxExample": 10
}
model.learn(game, learn_params, accept_model_params=accept_model_params)
model.save("networks/network_tic_tac_toe.h5")