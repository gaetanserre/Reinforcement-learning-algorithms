import numpy as np
import sys
import pandas as pd

class Agent:

  @staticmethod
  def get_table_value (table, action, state):
    return table[action][state] if state in table[action] else 0

  @staticmethod
  def create_table(nb_actions):
    table = []
    for _ in range(nb_actions):
      table.append({})
    return table
  
  @staticmethod
  def update_q_table(alpha, gamma, q_table, old_state, new_env, action, max_value=None):
    reward = new_env.get_reward()
    actual_value = Agent.get_table_value(q_table, action, old_state)
    try:
      max_list = list(map(lambda a: Agent.get_table_value(q_table, a, new_env.state), new_env.get_actions()))
      max_value = max(max_list)
    except:
      max_value = 0
    return (1-alpha) * actual_value + alpha * (reward + gamma * max_value)
  
  def __init__(self, create_env, Action, seed=None):
    np.random.seed(seed)
    self.create_env = create_env
    self.Action = Action

  # Greedy algorithm
  def choose_action(self, env, table, disp=False):
    actions = env.get_actions()
    values = list(map(lambda a: Agent.get_table_value(table, a, env.state), actions))

    if values.count(values[0])==len(values):
      if disp:
        print("random")
      return self.Action(np.random.choice(actions))
    else:
      idx_max = max(range(len(values)), key=values.__getitem__)
    return actions[idx_max]

  def greedy_fit(self, nb_iterations=2000):
    self.env = self.create_env()
    greedy_table = Agent.create_table(self.env.nb_actions)
    for _ in range(nb_iterations):
      action = self.choose_action(greedy_table)
      old_state = self.env.state
      self.env.set_state(self.env.get_new_state(action))
      reward = self.env.get_reward()
      greedy_table[action][old_state] = self.get_table_value(greedy_table, action, old_state) + reward

      if self.env.is_final_state():
        self.env = self.create_env()
    return greedy_table

  # Q-Learning algorithm
  def choose_q_action(self, env, q_table, gambling_rate):
    if np.random.random() <= gambling_rate:
      return self.Action(np.random.choice(env.get_actions()))
    else:
      return self.choose_action(env, q_table)
  
  def Q_fit(self, alpha=1, gamma=0.95, nb_iterations=4000):
    self.env = self.create_env()
    gambling_rate = 1

    q_table = Agent.create_table(self.env.nb_actions)
    for _ in range(nb_iterations):
      action = self.choose_q_action(self.env, q_table, gambling_rate)
      gambling_rate -= 1/nb_iterations
      old_state = self.env.state
      self.env.set_state(self.env.get_new_state(action))
      q_table[action][old_state] = Agent.update_q_table(alpha, gamma, q_table, old_state, self.env, action)

      if self.env.is_final_state():
        self.env = self.create_env()

    return q_table
  
  @staticmethod
  def inverse_player(player):
    return 1 if player == 0 else 0

  def Q_fit_2_players(self, alpha=0.9, gamma=0.95, nb_iterations=4000):
    self.env = self.create_env(1)
    self.env1 = self.create_env(1)
    self.env2 = self.create_env(2)
    gambling_rate = 1

    q_table1 = Agent.create_table(self.env1.nb_actions)
    q_table2 = Agent.create_table(self.env2.nb_actions)
    q_tables = [q_table1, q_table2]
    envs = [self.env1, self.env2]
    player = np.random.randint(0, 2)

    for _ in range(nb_iterations):
      action = self.choose_q_action(envs[player], q_tables[player], gambling_rate)

      gambling_rate -= 1/nb_iterations
      
      old_state = envs[player].state
      new_state = envs[player].get_new_state(action)
      envs[0].set_state(new_state)
      envs[1].set_state(new_state)

      q_tables[player][action][old_state] = Agent.update_q_table(
        alpha, gamma, q_tables[player], old_state, envs[player], action)

      if envs[player].is_final_state():
        envs[0] = self.create_env(1)
        envs[1] = self.create_env(2)
      player = Agent.inverse_player(player)
    return q_tables


  def policy_to_df(self, policy):
    return pd.DataFrame(policy, index=map(self.env.action_to_str, self.env.actions_list))