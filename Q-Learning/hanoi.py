import itertools
from enum import IntEnum
from gamei import Gamei

class Action(IntEnum):
  L2MID   = 0
  L2RIGHT = 1
  M2LEFT  = 2
  M2RIGHT = 3
  R2MID   = 4
  R2LEFT  = 5

class Hanoi(Gamei):

  @staticmethod
  def action_to_str(action):
    if action == 0: return "LEFT -> MID"
    if action == 1: return "LEFT -> RIGHT"
    if action == 2 : return "MID -> LEFT"
    if action == 3: return "MID -> RIGHT"
    if action == 4: return "RIGHT -> MID"
    if action == 5: return "RIGHT -> LEFT"  

    
  def __init__(self, N, state=None):
    if state is None:
      """
      A state is a tuple of size N. Each index corresponds
      to a disk, the first one is the smallest and the last one the biggest.
      Each element corresponds to the peg where the disk at index i is located.
      """
      super().__init__(tuple([0] * N), Action)
    else:
      super().__init__(state, Action)
    
    self.shape = (self.nb_actions, len(list(itertools.product(range(0,3), repeat=N))))
    self.rewards = {}
    for state in itertools.product(range(0,3), repeat=N):
      self.rewards[state] = -10
    self.final_state = tuple([2] * N)
    self.rewards[self.final_state] = 10

    self.moves = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 0)]
  
  def get_reward(self):
    return self.rewards[self.state]
  
  def disc_on_peg(self, peg):
    return [disc for disc in range(len(self.state)) if self.state[disc] == peg]

  def is_move_allowed(self, move):
    disc_from, disc_to = None, None
    disc_from = self.disc_on_peg(move[0])
    disc_to = self.disc_on_peg(move[1])

    if disc_from == []:
      return False
    else:
      disc_from = min(disc_from)
      return disc_to == [] or disc_from < min(disc_to)
  
  def get_actions(self):
    actions = []
    for move, action in zip(self.moves, self.actions_list):
      if self.is_move_allowed(move):
        actions.append(action)
    return actions
  
  def get_new_state(self, action):
    move = None
    for i in range(len(self.actions_list)):
      if action == self.actions_list[i]:
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

  @staticmethod
  def play(agent, policy, display=True):
    agent.env = agent.create_env()
    nb_moves = 0
    if display: print(agent.env.state)
    while agent.env.state != agent.env.final_state and nb_moves < 100:
      action = agent.choose_action(policy)
      agent.env.set_state(agent.env.get_new_state(action))
      nb_moves += 1
      if display:
        print(agent.env.action_to_str(action))
        print(agent.env.state)
        print(f"Number of moves: {nb_moves}")
    return nb_moves