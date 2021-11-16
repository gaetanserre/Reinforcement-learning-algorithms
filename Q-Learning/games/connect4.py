import numpy as np
from enum import IntEnum
from gamei import Gamei

ROWS = 3
COLUMNS = 4
MAX_VALUE = 9999

class Action(IntEnum):
  ONE     = 0
  TWO     = 1
  THREE   = 2
  FOUR    = 3

class Connect4(Gamei):

  @staticmethod
  def action_to_str(action):
    return f"{action + 1} column"
  
  def __init__(self, player, state=None):
    if state is None:
      """
      A state is a tuple of size N. Each index corresponds
      to a disk, the first one is the smallest and the last one the biggest.
      Each element corresponds to the peg where the disk at index i is located.
      """
      super().__init__(tuple(np.zeros((ROWS, COLUMNS)).flatten()), Action)
    else:
      super().__init__(state, Action)
    
    self.player = player

  def convert_state(self):
    return np.array(list(self.state)).reshape((ROWS, COLUMNS))

  def winning_position(self, player):
    state = self.convert_state()
    # Check horizontal locations for win
    for c in range(COLUMNS-3):
      for r in range(ROWS):
        if state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player and state[r][c+3] == player:
          return True

    # Check vertical locations for win
    for c in range(COLUMNS):
      for r in range(ROWS-3):
        if state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player and state[r+3][c] == player:
          return True

    # Check positively sloped diagonals
    for c in range(COLUMNS-3):
      for r in range(ROWS-3):
        if state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player and state[r+3][c+3] == player:
          return True

    # Check negatively sloped diagonals
    for c in range(COLUMNS-3):
      for r in range(3, ROWS):
        if state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player and state[r-3][c+3] == player:
          return True
    return False
    
  def get_reward(self):
    if self.winning_position(self.player): return MAX_VALUE

    player = 2 if self.player == 1 else 1
    if self.winning_position(player): return -MAX_VALUE
    elif self.get_actions() == []:
      return 0
    else: return -10
  
  def get_actions(self):
    state = self.convert_state()
    actions = []
    for action in self.actions_list:
      if state[0][action] == 0:
        actions.append(action)
    return actions
  
  def get_new_state(self, action):
    state = self.convert_state()
    for r in range(ROWS-1, -1, -1):
      if state[r][action] == 0:
        state[r][action] = self.player
        return tuple(state.flatten())

  def is_final_state(self):
    reward = self.get_reward()
    return abs(reward) == MAX_VALUE or reward == 0
  