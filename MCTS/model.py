import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mcts import MCTS
from arena import Arena

class Model:
  def __init__(self, nnet, summary=False):
    self.model = nnet
    losses = {"policy": "categorical_crossentropy", "value": "mean_squared_error"}
    metrics = {"policy": "accuracy", "value": "mean_squared_error"}
    self.model.compile(loss=losses, optimizer="adam", metrics=metrics)
    if summary:
      self.model.summary()
  
  def predict(self, state):
    data = np.expand_dims(state, axis=0)
    pred = self.model.predict(data)
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
      train_positions.append(state)
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
        return train_positions, train_policies, train_values


  def train(self, game, nb_iter, nb_simulations, nb_games, nb_epochs, accept_model_params, plot=False):
    accept_params = accept_model_params
    train_positions, train_policies, train_values = [], [], []
    for i in range(nb_iter):
      print(f"{(i+1)}/{nb_iter}...")
      for _ in range(nb_games):
        train_positions_t, train_policies_t, train_values_t = self.execute_episode(
          game, nb_simulations)
        train_positions.extend(train_positions_t)
        train_policies.extend(train_policies_t)
        train_values.extend(train_values_t)
      
      train_positions = np.array(train_positions)
      train_policies = np.array(train_policies)
      train_values = np.array(train_values)

      if plot:
        plt.hist(train_values)
        plt.title("Values distribution")
        plt.show()
      
      target = {"policy": train_policies, "value": train_values}
      old_model = Model(tf.keras.models.clone_model(self.model))
      history = self.model.fit(train_positions, target, verbose=0, epochs=nb_epochs)

      if accept_params["enable"]:
        arena = Arena(self, old_model, game)
        w, l = arena.play_games(accept_params["nb_games"], accept_params["nb_simulations"])
        win_ratio = 0 if w+l == 0 else w / (w+l)
        print(w, l)
        print(f"Win ratio: {win_ratio}")
        if win_ratio < accept_params["min_win_ratio"]:
          print("Reject model.")
          self.model = old_model.model
          train_positions = train_positions.tolist()
          train_policies = train_policies.tolist()
          train_values = train_values.tolist()
        else:
          print("Accept model.")
          train_positions, train_policies, train_values = [], [], []

      print("Done")
      policy_acc = history.history["policy_accuracy"][-1]
      value_mse = history.history["value_mean_squared_error"][-1]
      print(f"policy_accuracy: {policy_acc: .2f} value_mse: {value_mse: .2f}")

  def save(self, path):
    self.model.save(path)
  
  def load(self, path):
    self.model = tf.keras.models.load_model(path)