import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from tqdm import tqdm
from multiprocessing import current_process

from lib.mcts import MCTS
from lib.arena import Arena

def unpack(l):
  res = []
  for i in l:
    for j in i:
      res.append(j)
  return res


class Model:
  def __init__(self, nnet, summary=False):
    self.nnet = nnet
    if summary:
      self.nnet.summary()
    
    self.train_examples_history = []
  
  def predict(self, state):
    data = np.expand_dims(state, axis=0)
    pred = self.nnet.predict(data)
    return pred[0].flatten(), pred[1].flatten()[0]
  

  def execute_episode(self, game, nb_simulations):
    train_positions = []
    train_policies = []
    train_values  = []
    state = game.get_init_state()
    
    max_move_temp = 5
    nb_moves = 0
    while True:
      nb_moves += 1
      train_positions.append(game.get_canonical_form(state))
      mcts = MCTS(game, state, self, nb_simulations)
      root = mcts.run()

      temperature = int(nb_moves < max_move_temp)
      policy, action = root.select_action(temperature, game.nb_actions)
      train_policies.append(policy)
      
      state = game.get_new_state(state, action)
      player = game.get_player(state)
      reward = game.get_reward(state)

      if reward is not None:
        for i in range(len(train_positions)):
          p_player = (-1)**i
          train_values.append(reward if player == p_player else -reward)
        return [(train_positions, train_policies, train_values)]
  
  def train(self, train_examples, nb_epochs):
    train_positions, train_policies, train_values = list(zip(*train_examples))
    train_positions = unpack(train_positions)
    train_policies = unpack(train_policies)
    train_values = unpack(train_values)

    train_positions = np.asarray(train_positions)
    train_policies = np.asarray(train_policies)
    train_values = np.asarray(train_values)
    target = {"policy": train_policies, "value": train_values}
    return self.nnet.fit(train_positions, target, verbose=1, epochs=nb_epochs)
  
  def learn(self, game, learn_params, accept_model_params):
    nb_iter = learn_params["nb_iter"]
    nb_games = learn_params["nb_games"]
    nb_simulations = learn_params["nb_simulations"]
    nb_epochs = learn_params["nb_epochs"]
    maxExample = learn_params["maxExample"]

    accept_params = accept_model_params
    for i in range(nb_iter):
      print(f"{(i+1)}/{nb_iter}...")

      # Status bar configuration
      current = current_process()
      pos = current._identity[0]-1 if len(current._identity) > 0 else 0
      pbar = tqdm(total=nb_games, desc="Self play", position=pos)

      train_examples = []
      for _ in range(nb_games):
        train_examples += self.execute_episode(game, nb_simulations)
        pbar.update(1)

      self.train_examples_history.append(train_examples)
      if len(self.train_examples_history) > maxExample:
        self.train_examples_history.pop(0)
      
      train_examples = []
      for e in self.train_examples_history:
        train_examples.extend(e)
      shuffle(train_examples)

      self.save("tmp/old_nn.h5")
      old_model = Model(None)
      old_model.load("tmp/old_nn.h5")
      history = self.train(train_examples, nb_epochs)

      arena = Arena(self, old_model, game)
      w, l = arena.play_games(accept_params["nb_games"], accept_params["nb_simulations"])
      win_ratio = 0 if w+l == 0 else w / (w+l)
      print(w, l)
      print(f"Win ratio: {win_ratio: .2f}")
      if win_ratio < accept_params["min_win_ratio"]:
        print("Reject model.")
        self.load("tmp/old_nn.h5")
      else:
        print("Accept model.")

      print("Done")
      policy_acc = history.history["policy_accuracy"][-1]
      value_mse = history.history["value_mean_squared_error"][-1]
      print(f"policy_accuracy: {policy_acc: .2f} value_mse: {value_mse: .2f}")

  def save(self, path):
    self.nnet.save(path)
  
  def load(self, path):
    self.nnet = tf.keras.models.load_model(path)
