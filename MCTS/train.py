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
params = {"nb_games": 20, "nb_simulations": 25, "min_win_ratio": 0.5}
model.train(game, nb_iter=30, nb_simulations=25, nb_games=100, nb_epochs=20, accept_model_params=params, plot=False)
model.save("networks/network_tic_tac_toe.h5")