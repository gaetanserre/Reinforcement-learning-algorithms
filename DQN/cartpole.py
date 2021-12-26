from agent import DQNAgent
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

dqn_agent = DQNAgent(env, np.array([0, 1]))
dqn_agent.learn(env, "DQN_NN", nb_episodes=300)
dqn_agent.dqn.save_nn("DQN_NN")


obs = env.reset()
done = False
steps = 0

while not done:
  steps += 1
  plt.imshow(env.render(mode="rgb_array"))
  obs, _, done, _ = env.step(dqn_agent.act(obs))

env.close()

print(f"Total steps survived: {steps}")