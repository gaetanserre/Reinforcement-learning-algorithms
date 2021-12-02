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
params = {"enable": False, "nb_games": 15, "nb_simulations": 50, "min_win_ratio": 0.55}
model.train(game, nb_iter=40, nb_simulations=25, nb_games=150, nb_epochs=40, accept_model_params=params, plot=True)
model.save("networks/network_tic_tac_toe.h5")


