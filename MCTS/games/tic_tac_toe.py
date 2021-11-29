import numpy as np
import matplotlib.pyplot as plt
from mcts import MCTS


class TicTacToe():
  ROWS = 3
  COLUMNS = 3
  nb_actions = COLUMNS * ROWS

  @staticmethod
  def get_player(state):
    return state[0,0,1]

  @staticmethod
  def get_init_state():
    init_state = np.zeros((TicTacToe.ROWS, TicTacToe.COLUMNS, 2))
    init_state[:, :, 1] = 1
    return init_state

  def __init__(self):
    pass

  def get_actions(self, state):
    actions = np.zeros(TicTacToe.nb_actions)
    for i in range(TicTacToe.ROWS):
      for j in range(TicTacToe.COLUMNS):
        if state[i, j, 0] == 0:
          square = (i * TicTacToe.COLUMNS) + j
          actions[square] = 1
    return actions
  
  def get_new_state(self, state, action):
    state = state.copy()
    player = TicTacToe.get_player(state)
    row = action // TicTacToe.COLUMNS
    column = action % 3
    state[row, column, 0] = player
    state[:, :, 1] = -player
    return state
  
  def is_win(self, state, player):
    state = state.copy()
    state = state[:, :, 0]
    # Check horizontal locations for win
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(TicTacToe.ROWS):
        if state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player:
          return True

    # Check vertical locations for win
    for c in range(TicTacToe.COLUMNS):
      for r in range(TicTacToe.ROWS-2):
        if state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player:
          return True

    # Check positively sloped diagonals
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(TicTacToe.ROWS-2):
        if state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player:
          return True

    # Check negatively sloped diagonals
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(2, TicTacToe.ROWS):
        if state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player:
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
    color = np.zeros((TicTacToe.ROWS, TicTacToe.COLUMNS, 3))
    color[state == 1] = [1, 0, 0]
    color[state == -1] = [1, 1, 0]
    return color
  
  @staticmethod
  def play_vs(model, state, nb_simul):
    game = TicTacToe()
    while game.get_reward(state) is None:
      mcts = MCTS(TicTacToe(), state, model, nb_simul)
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
