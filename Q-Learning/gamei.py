class Gamei:
  nb_actions = None

  def __init__(self, state):
    self.state = state

  def get_reward(self):
    pass

  def get_actions(self):
    pass

  def get_new_state(self, action):
    pass
  
  def is_final_state(self):
    pass

  def set_state(self, state):
    self.state = state