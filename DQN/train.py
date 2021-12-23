from cartpole import DQNAgent
import gym

env = gym.make("CartPole-v1")

dqn_agent = DQNAgent(env)
dqn_agent.learn(env, "DQN_NN", nb_episodes=300)
