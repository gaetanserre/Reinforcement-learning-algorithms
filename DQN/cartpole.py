from grid2op.Agent import BaseAgent
import gym
from DenseNN import DenseNN
from DQN import DQN
import numpy as np

env = gym.make("CartPole-v1")


class DQNAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, env, curr_dir):
      """Initialize a new agent."""
      BaseAgent.__init__(self, action_space=env.action_space)
      self.all_actions = np.array([0, 1])
      """
      self.all_actions = [env.action_space({})]
      actions = np.load(os.path.join(curr_dir, "top1000_actions.npz"), allow_pickle=True)["actions"]
      for action in actions:
        self.all_actions.append(env.action_space.from_vect(action))
      self.all_actions = np.asarray(self.all_actions)
      """
      self.dqn = DQN(self, (lambda : DenseNN(4, self.all_actions.shape[0])))

    def act(self, observation, reward, done):
      """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
      # do nothing for example (with the empty dictionary) :
      
      return self.dqn.select_action(observation)

    def train(self, env, nb_episode):
      self.dqn.replay_exp(env, nb_episode=nb_episode)
    


dqn_agent = DQNAgent(env, ".")
dqn_agent.train(env, 300)


