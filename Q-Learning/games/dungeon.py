import numpy as np
from enum import IntEnum
from gamei import Gamei

class Action(IntEnum):
  FORWARD = 0
  BACKWARD = 1

class Dungeon(Gamei):
  @staticmethod
  def action_to_str(action):
    if action == 0: return "FORWARD"
    else: return "BACKWARD"

  def __init__(self, state=0):
    super().__init__(state, Action)
    self.rewards = [2, 0, 0, 0, 10]
    self.state = state
  
  def get_reward(self):
    return self.rewards[self.state]

  def get_actions(self):
    return self.actions_list
  
  def get_new_state(self, action):
    wind = np.random.random() <= 0.1

    if wind:
      action = Action(int(not action))
    
    if action == Action.FORWARD:
      return min(self.state + 1, len(self.rewards) - 1)
    else: return 0
  
  def set_state(self, state):
    self.state = state
      
  def is_final_state(self):
    return False