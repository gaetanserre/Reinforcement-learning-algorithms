import itertools
from enum import IntEnum

class Action(IntEnum):
  R2LEFT  = 0
  R2MID   = 1
  L2RIGHT = 2
  L2MID   = 3
  M2RIGHT = 4
  M2LEFT  = 5

class Hanoi():
  nb_actions = 6
  def __init__(self, N, state=None):
    if state is None:
      self.state = tuple([0] * N)
    else:
      self.state = state
    
    self.shape = (Hanoi.nb_actions, len(list(itertools.product(range(0,3), repeat=N))))
    self.rewards = {}
    for state in itertools.product(range(0,3), repeat=N):
      self.rewards[state] = -10
    self.final_state = tuple([2] * N)
    self.rewards[self.final_state] = 10

    self.moves = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    self.actions = [Action.L2MID, Action.L2RIGHT, Action.M2LEFT, Action.M2RIGHT, Action.R2LEFT, Action.R2MID]
  
  def get_reward(self):
    return self.rewards[self.state]
  
  def disc_on_pep(self, pep):
    return [disc for disc in range(len(self.state)) if self.state[disc] == pep]

  def is_move_allowed(self, move):
    disc_from, disc_to = None, None
    disc_from = self.disc_on_pep(move[0])
    disc_to = self.disc_on_pep(move[1])

    if disc_from == []:
      return False
    
    else:
      disc_from = min(disc_from)
      return disc_to == [] or disc_from < min(disc_to)
  
  def get_actions(self):
    actions = []
    for move, action in zip(self.moves, self.actions):
      if self.is_move_allowed(move):
        actions.append(action)
    return actions
  
  def get_new_state(self, action):
    move = None
    for i in range(len(self.actions)):
      if action == self.actions[i]:
        move = self.moves[i]
        break
    
    if move is None: return self.state
    
    new_state = list(self.state)
    disc_to_move = min([disc for disc in range(len(self.state)) if self.state[disc] == move[0]])
    new_state[disc_to_move] = move[1]
    return tuple(new_state)
  
  def set_state(self, state):
    self.state = state
  
  def is_final_state(self):
    return self.state == self.final_state