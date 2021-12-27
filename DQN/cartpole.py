from agent import DQNAgent
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
network_dir = "DQN_NN"

# Create the agent
dqn_agent = DQNAgent(env, np.array([0, 1]))

# Train the agent
dqn_agent.learn(env, network_dir, nb_episodes=500)
dqn_agent.dqn.save_nn(network_dir)

# Test the agent
dqn_agent.dqn.load_nn(network_dir)
obs = env.reset()
done = False
steps = 0
while not done:
  steps += 1
  plt.imshow(env.render(mode="rgb_array"))
  obs, _, done, _ = env.step(dqn_agent.act(obs))

env.close()

print(f"Total steps survived: {steps}")