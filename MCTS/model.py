import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
from mcts import MCTS
import matplotlib.pyplot as plt


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_model(input_shape, nb_actions):
  inputs = layers.Input(input_shape)
  x_old = layers.Conv2D(10, (1, 1))(inputs)
  x_old = layers.ReLU()(x_old)

  x = layers.Conv2D(10, (1, 1))(x_old)
  x = layers.ReLU()(x)
  x = layers.Conv2D(10, (1, 1))(x)
  x = layers.ReLU()(x)
  x_old = layers.Add()([x, x_old])

  x = layers.Conv2D(10, (1, 1))(x_old)
  x = layers.ReLU()(x)
  x = layers.Conv2D(10, (1, 1))(x)
  x = layers.ReLU()(x)
  x = layers.Add()([x, x_old])

  outputs = layers.Flatten()(x)

  policy = layers.Dense(nb_actions, activation="softmax", name="policy")(outputs)
  value = layers.Dense(1, activation="tanh", name="value")(outputs)
  return tf.keras.Model(inputs=inputs, outputs=[policy, value])

class Model:
  def __init__(self, input_shape, nb_actions, summary=False):
    self.input_shape = input_shape
    self.model = create_model(input_shape, nb_actions)
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

    while True:
      mcts = MCTS(game, state, self, nb_simulations)
      root = mcts.run()

      train_positions.append(state)
      policy = [0] * game.nb_actions
      for a, v in root.children.items():
        policy[a] = v.visit_count
      policy /= np.sum(policy)
      train_policies.append(policy)

      action = root.select_action()
      state = game.get_new_state(state, action)
      reward = game.get_reward(state)

      if reward is not None:
        for _ in train_positions:
          train_values.append(reward)
        return train_positions, train_policies, train_values


  def train(self, game, nb_iter, nb_simulations, nb_games, nb_epochs):
    for i in range(nb_iter):
      print(f"{(i+1)}/{nb_iter}...", end="")
      train_positions, train_policies, train_values = [], [], []
      for _ in range(nb_games):
        train_positions_t, train_policies_t, train_values_t = self.execute_episode(
          game, nb_simulations)
        train_positions = train_positions + train_positions_t
        train_policies = train_policies + train_policies_t
        train_values = train_values + train_values_t
      
      train_positions = np.array(train_positions)
      train_policies = np.array(train_policies)
      train_values = np.array(train_values)
      plt.hist(train_values)
      plt.title("Values distribution")
      plt.show()
      
      target = {"policy": train_policies, "value": train_values}
      history = self.model.fit(train_positions, target, verbose=0, epochs=nb_epochs)
      print("Done")
      policy_acc = history.history["policy_accuracy"][-1]
      value_mse = history.history["value_mean_squared_error"][-1]
      print(f"policy_accuracy: {policy_acc: .2f} value_mse: {value_mse: .2f}")

  def save(self, path):
    self.model.save(path)
  
  def load(self, path):
    self.model = tf.keras.models.load_model(path)