from mcts import MCTS

import numpy as np
import matplotlib.pyplot as plt

class GameI:

  @staticmethod
  def get_player(state):
    pass

  @staticmethod
  def get_canonical_form(state):
    pass
  
  @staticmethod
  def get_init_state():
    pass

  def get_actions(self, state):
    pass

  def __init__(self):
    pass
  

  def get_new_state(self, state, action):
    pass

  def get_reward(self, state):
    pass
  
  def colorize_state(self, state):
    pass

  def play_vs(self, model, state, nb_simul):
    game = self
    while game.get_reward(state) is None:
      mcts = MCTS(game, state, model, nb_simul)
      root = mcts.run()
      
      _, action = root.select_action(0, game.nb_actions)
      o_state = state.copy()
      state = game.get_new_state(state, action)
      plt.imshow(game.colorize_state(state))
      plt.show()
      print(f"Model prediction: {model.predict(game.get_canonical_form(o_state))}")
      print(f"Root value: {root.value()}")

      if game.get_reward(state) is not None:
        break

      action = int(input("Column:"))
      o_state = state.copy()
      state = game.get_new_state(state, action)
      plt.imshow(game.colorize_state(state))
      plt.show()
      print(f"Model prediction: {model.predict(game.get_canonical_form(o_state))}")
  
  def play_vs_bot(self, model1, model2, state, nb_simul):
    current_player = model1

    while self.get_reward(state) is None:
      mcts = MCTS(self, state, current_player, nb_simul)
      root = mcts.run()
      _, action = root.select_action(0, self.nb_actions)
      state = self.get_new_state(state, action)
      plt.imshow(self.colorize_state(state))
      plt.show()
      current_player = model2 if current_player == model1 else model2