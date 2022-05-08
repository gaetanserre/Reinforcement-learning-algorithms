import numpy as np
from games.gamei import GameI

class Connect2(GameI):
  nb_actions = 4
  shape = (nb_actions, )

  @staticmethod
  def get_player(state):
    return state[4]
  
  @staticmethod
  def get_canonical_form(state):
    res = state[:-1].copy()
    return res * Connect2.get_player(state)

  @staticmethod
  def get_init_state():
    init_state = np.zeros(5)
    init_state[4] = 1
    return init_state

  def __init__(self):
    self.win = 2
    
  
  def get_actions(self, state):
    actions = np.zeros(4, dtype=np.int)
    for i, s in enumerate(state):
      if s == 0:
        actions[i] = 1
    return actions
  
  def get_new_state(self, state, action):
    player = Connect2.get_player(state)
    new_state = state.copy()
    new_state[action] = player
    new_state[4] *= -1
    return new_state
  
  def is_win(self, state, player):
    count = 0
    for index in range(state.shape[0]-1):
        if state[index] == player:
            count = count + 1
        else:
            count = 0
        if count == self.win:
            return True
    return False

  def get_reward(self, state):
    player = self.get_player(state)
    if self.is_win(state, player):
      return 1
    elif self.is_win(state, -player):
      return -1
    elif not self.get_actions(state).any():
      return 0
    else:
      return None
  
  def colorize_state(self, state):
    state = state.copy()
    state = state[:4]
    color = np.zeros((1, Connect2.nb_actions, 3))
    color[:, state == 1] = [1, 0, 0]
    color[:, state == -1] = [1, 1, 0]
    return color