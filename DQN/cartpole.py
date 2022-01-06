from agent import DQNAgent
from env import Env
import numpy as np
import matplotlib.pyplot as plt

env = Env()
network_dir = "DQN_NN"

# Create the agent
dqn_agent = DQNAgent(env, np.array([0, 1]))

# Train the agent
dqn_agent.learn(env, network_dir, nb_episodes=500)
dqn_agent.dqn.save_nn(network_dir)

# Test the agent
dqn_agent.dqn.load_nn(network_dir)

total_rewards = 0
nb_iterations = 100
for i in range(nb_iterations):
  obs = env.reset()
  done = False
  steps = 0
  while not done:
    steps += 1
    plt.imshow(env.render(mode="rgb_array"))
    action = dqn_agent.act(obs)
    obs, reward, done, _ = env.step(action)
    total_rewards += reward
  print(f"({i+1}/{nb_iterations}) Total steps survived: {steps}")
  
env.close()
avg_reward = total_rewards / nb_iterations
print(f"Average reward: {avg_reward}")
if avg_reward >= 196:
  print("Problem solved.")
