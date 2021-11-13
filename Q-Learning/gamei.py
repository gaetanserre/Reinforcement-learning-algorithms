class Gamei:
  
  def __init__(self, state, Action):
    self.state = state
    self.actions_list = [action for action in Action]
    self.nb_actions = len(self.actions_list)

  @staticmethod
  def action_to_str(action):
    pass      

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