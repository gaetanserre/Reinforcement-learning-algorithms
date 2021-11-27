import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def create_model(input_shape, nb_actions):
  inputs = layers.Input(input_shape)
  outputs = layers.Dense(64, activation="relu")(inputs)
  policy = layers.Dense(nb_actions, activation="softmax", name="policy")(outputs)
  value = layers.Dense(1, activation="linear", name="value")(outputs)
  return tf.keras.Model(inputs=inputs, outputs=[policy, value])

class Model:
  def __init__(self, input_shape, nb_actions):
    self.input_shape = input_shape
    self.model = create_model(input_shape, nb_actions)
    losses = {"policy": "categorical_crossentropy", "value": "means_squared_error"}
    self.model.compile(loss=losses, optimizer="adam")
  
  def predict(self, state, player):
    player = np.array([player])
    data = np.concatenate((state, player), axis=0).reshape(1, 5)
    print(data.shape)
    return self.model.predict(data)

if __name__ == "__main__":
  model = Model((5,), 4)
  print(model.predict(np.zeros((4)), 12))