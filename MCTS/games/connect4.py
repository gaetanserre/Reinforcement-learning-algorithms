import numpy as np
import matplotlib.pyplot as plt
from mcts import MCTS


class Connect4():
  ROWS = 6
  COLUMNS = 7
  nb_actions = COLUMNS

  @staticmethod
  def get_player(state):
    return state[0,0,1]

  @staticmethod
  def get_init_state():
    init_state = np.zeros((Connect4.ROWS, Connect4.COLUMNS, 2))
    init_state[:, :, 1] = 1
    return init_state

  def __init__(self):
    pass

  def get_actions(self, state):
    actions = np.zeros(Connect4.COLUMNS)
    for action in range(Connect4.COLUMNS):
      if state[0, action, 0] == 0:
        actions[action] = 1
    return actions
  
  def get_new_state(self, state, action):
    state = state.copy()
    player = Connect4.get_player(state)
    for r in range(Connect4.ROWS-1, -1, -1):
      if state[r, action, 0] == 0:
        state[r, action, 0] = player
        state[:, :, 1] = -player
        return state
  
  def is_win(self, state, player):
    state = state.copy()
    state = state[:, :, 0]
    # Check horizontal locations for win
    for c in range(Connect4.COLUMNS-3):
      for r in range(Connect4.ROWS):
        if state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player and state[r][c+3] == player:
          return True

    # Check vertical locations for win
    for c in range(Connect4.COLUMNS):
      for r in range(Connect4.ROWS-3):
        if state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player and state[r+3][c] == player:
          return True

    # Check positively sloped diagonals
    for c in range(Connect4.COLUMNS-3):
      for r in range(Connect4.ROWS-3):
        if state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player and state[r+3][c+3] == player:
          return True

    # Check negatively sloped diagonals
    for c in range(Connect4.COLUMNS-3):
      for r in range(3, Connect4.ROWS):
        if state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player and state[r-3][c+3] == player:
          return True
    return False

  def get_reward(self, state):
    if self.is_win(state, 1):
      return 1
    elif self.is_win(state, -1):
      return -1
    elif not self.get_actions(state).any():
      return 0
    else:
      return None
  
  def colorize_state(self, state):
    state = state.copy()
    state = state[:, :, 0]
    color = np.zeros((Connect4.ROWS, Connect4.COLUMNS, 3))
    color[state == 1] = [1, 0, 0]
    color[state == -1] = [1, 1, 0]
    return color
  
  @staticmethod
  def play_vs(model, state, nb_simul):
    game = Connect4()
    while game.get_reward(state) is None:
      mcts = MCTS(Connect4(), state, model, nb_simul)
      root = mcts.run()
      state = root.select_child().state
      plt.imshow(game.colorize_state(state))
      plt.show()
      print(f"Eval: {model.predict(state)}")

      if game.get_reward(state) is not None:
        break

      action = int(input("Column:")) - 1
      state = game.get_new_state(state, action)
      plt.imshow(game.colorize_state(state))
      plt.show()
      print(f"Eval: {model.predict(state)}")