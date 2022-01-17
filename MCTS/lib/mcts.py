import numpy as np

def ucb_score(parent, child):
  prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
  if child.visit_count > 0:
      value_score = -child.value()
  else:
      value_score = 0

  return value_score + prior_score

class Node:
  def __init__(self, prior, player, state):
    self.prior = prior
    self.visit_count = 0
    self.value_sum = 0
    self.children = {}
    self.player = player
    self.state = state

  def value(self):
    return self.value_sum / self.visit_count

  def select_action(self, temperature, nb_actions):
    counts = np.zeros(nb_actions)
    for a, child in self.children.items():
      counts[a] = child.visit_count
    
    if temperature == 0:
      probs = np.zeros(counts.shape[0])
      bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
      bestA = np.random.choice(bestAs)
      probs[bestA] = 1
      return probs, bestA
    else:
      counts = counts ** (1/temperature)
      visit_counts_distribution = counts / np.sum(counts)
      actions = np.arange(0, nb_actions, 1)
      bestA = np.random.choice(actions, p=visit_counts_distribution)
      return visit_counts_distribution, bestA
  
  def select_child(self):
    best_score = -np.inf
    best_child = None

    for child in self.children.values():
      score = ucb_score(self, child)
      if score > best_score:
        best_score = score
        best_child = child

    return best_child
  
  def expand(self, policy, game):
    for a, prob in enumerate(policy):
      if prob != 0:
        child_state = game.get_new_state(self.state, a)
        self.children[a] = Node(prob, -self.player, child_state)


class MCTS:
  def __init__(self, game, state, model, nb_simulations):
    self.state = state
    self.game = game
    self.player = game.get_player(state)
    self.model = model
    self.nb_simulations = nb_simulations
  
  def backtrack(self, visited, player, reward):
    for node in visited:
      node.value_sum += reward if player == node.player else -reward
      node.visit_count += 1

  @staticmethod
  def normalize_policy(policy, actions):
    res = policy * actions
    sum_pol = np.sum(res)
    if sum_pol > 0:
      res /= sum_pol
    return res
  
  def run(self):
    root = Node(0, self.player, self.state)
    policy, _ = self.model.predict(self.game.get_canonical_form(root.state))
    policy = self.normalize_policy(policy, self.game.get_actions(root.state))
    root.expand(policy, self.game)

    for _ in range(self.nb_simulations):
      node = root
      state = root.state
      visited = [node]
      while len(node.children) > 0:
        node = node.select_child()
        visited.append(node)
      state = node.state
      reward = self.game.get_reward(state)
      if reward is None:
        policy, reward = self.model.predict(self.game.get_canonical_form(state))
        policy = self.normalize_policy(policy, self.game.get_actions(state))
        node.expand(policy, self.game)
      self.backtrack(visited, node.player, reward)

    return root
