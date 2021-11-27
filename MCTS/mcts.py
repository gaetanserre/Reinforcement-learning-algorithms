import numpy as np

def ucb_score(parent, child):
  def log(n):
    if n == 0: return 0
    else: return np.log(n)

  prior_score = child.prior * np.sqrt(log(parent.visit_count)) / (child.visit_count + 1)
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
    visit_counts = np.array([child.visit_count for child in self.children.values()])
    actions = [action for action in self.children.keys()]
    action = actions[np.argmax(visit_counts)]
    return action
  
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
        child_state = game.get_new_state(a)
        self.children[a] = Node(prob, -self.player, child_state)


class MCTS:
  def __init__(self, game, model, nb_simulations):
    self.game = game
    self.model = model
    self.nb_simulations = nb_simulations
  
  def backtrack(self, visited, reward):
    for node in visited:
      node.value_sum += reward * node.player
      node.visit_count += 1
  
  def run(self):
    root = Node(0, self.game.player, self.game.state)
    policy = self.model.policy(root.state, root.player)
    policy = policy * self.game.get_actions()
    policy /= np.sum(policy)
    root.expand(policy, self.game)

    for _ in range(self.nb_simulations):
      node = root
      game = self.game
      visited = [node]
      while len(node.children) > 0:
        node = node.select_child()
        visited.append(node)
      game.state = node.state
      game.player = node.player
      reward = game.get_reward()
      if reward is None:
        reward = self.model.reward(node.state, node.player)
        print(reward, node.state, node.player)
        policy = self.model.policy(node.state, node.player)
        policy = policy * self.game.get_actions()
        policy /= np.sum(policy)
        node.expand(policy, game)
      self.backtrack(visited, reward)

    return root
