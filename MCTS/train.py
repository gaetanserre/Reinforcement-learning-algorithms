# %%
from games.tic_tac_toe import TicTacToe
from networks.tic_tac_toe_net import create_network

from lib.model import Model
import numpy as np

np.random.seed(42)

# %%
game = TicTacToe()
model = Model(create_network(game.shape, game.nb_actions), summary=True)

# %%
accept_model_params = {"nb_games": 0, "nb_simulations": 25, "min_win_ratio": 0}
learn_params = {
  "nb_iter": 5,
  "nb_simulations": 25,
  "nb_games": 50,
  "nb_epochs": 40,
  "maxExample": 10
}
model.learn(game, learn_params, accept_model_params=accept_model_params)
model.save("networks/trained/network_tic_tac_toe.h5")