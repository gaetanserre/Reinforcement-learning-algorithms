import numpy as np

class Connect2():
  def __init__(self, player, state=None):
    if state is None:
      self.state = np.zeros(4)
    else:
      self.state = state
    
    self.player = player
    self.win = 2
    
  
  def get_actions(self):
    actions = np.zeros(4, dtype=np.int)
    for i, s in enumerate(self.state):
      if s == 0:
        actions[i] = 1
    return actions
  
  def get_new_state(self, action):
    new_state = self.state.copy()
    new_state[action] = self.player
    return new_state
  
  def is_win(self, player):
    count = 0
    for index in range(self.state.shape[0]):
        if self.state[index] == player:
            count = count + 1
        else:
            count = 0
        if count == self.win:
            return True
    return False

  def get_reward(self):
    if self.is_win(1):
      return 1
    elif self.is_win(-1):
      return -1
    elif self.get_actions() == []:
      return 0
    else:
      return None