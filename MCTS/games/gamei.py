import matplotlib.pyplot as plt
from mcts import MCTS

class GameI:

  @staticmethod
  def get_player(state):
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
      print(f"Model prediction: {model.predict(o_state)}")
      print(f"Root value: {root.value()}")

      if game.get_reward(state) is not None:
        break

      action = int(input("Column:"))
      o_state = state.copy()
      state = game.get_new_state(state, action)
      plt.imshow(game.colorize_state(state))
      plt.show()
      print(f"Model prediction: {model.predict(o_state)}")