# %%
import sys
sys.path.insert(1, "games/")
sys.path.insert(1, "networks/")
from tic_tac_toe import TicTacToe
from tic_tac_toe_net import create_network

from model import Model
import numpy as np

np.random.seed(42)

# %%
game = TicTacToe()
model = Model(create_network(game.shape, game.nb_actions), summary=True)

# %%
params = {"nb_games": 20, "nb_simulations": 75, "min_win_ratio": 0.51}
model.train(game, nb_iter=30, nb_simulations=50, nb_games=100, nb_epochs=40, accept_model_params=params)
model.save("networks/network_tic_tac_toe.h5")