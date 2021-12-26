from DenseNN import DenseNN
from DQN import DQN

class DQNAgent():
  """
  The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
  grid2op.Agent.BaseAgent.
  """
  def __init__(self, env, all_actions):
    """Initialize a new agent."""
    self.all_actions = all_actions
    self.dqn = DQN(self, (lambda : DenseNN(env.reset().shape[0], self.all_actions.shape[0])))

  def act(self, observation):
    """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
    # do nothing for example (with the empty dictionary) :
    
    return self.dqn.select_action(observation)

  def learn(self, env, path, nb_episodes):
    self.dqn.replay_exp(env, path, nb_episodes=nb_episodes)
  