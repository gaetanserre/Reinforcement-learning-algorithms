import numpy as np

def ucb_score(parent, child):
  prior_score = child.prior * np.sqrt(np.log(parent.visit_count + 1)) / (child.visit_count + 1)
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

  def select_action(self):
    visit_counts_distribution = np.array(
      [child.visit_count for child in self.children.values()],
      dtype=np.float32)

    visit_counts_distribution /= np.sum(visit_counts_distribution)
    actions = [action for action in self.children.keys()]
    return np.random.choice(actions, p=visit_counts_distribution)
  
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
  
  def backtrack(self, visited, reward):
    for node in visited:
      node.value_sum += reward * node.player
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
    policy, _ = self.model.predict(root.state)
    policy = self.normalize_policy(policy, self.game.get_actions(self.state))
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
        policy, reward = self.model.predict(node.state)
        policy = self.normalize_policy(policy, self.game.get_actions(state))
        node.expand(policy, self.game)
      self.backtrack(visited, reward)

    return root
