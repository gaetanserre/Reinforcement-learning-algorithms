#import tensorflow as tf
import torch
import torch.nn as nn
from utils import RANDOM_SEED
from torch_model_wrapper import TorchWrapper
torch.manual_seed(RANDOM_SEED)

#tf.random.set_seed(RANDOM_SEED)

"""
class DenseNN:
  def __init__(self, input_shape, nb_actions):
    learning_rate = 0.001
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Dense(24, input_shape=(input_shape,), activation="relu"))
    self.model.add(tf.keras.layers.Dense(12, input_shape=(input_shape,), activation="relu"))
    self.model.add(tf.keras.layers.Dense(nb_actions, activation="linear"))
    self.model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

  def predict(self, X):
    return self.model.predict(X)

  def fit(self, X, y, batch_size, shuffle):
    self.model.fit(X, y, verbose=0, batch_size=batch_size, shuffle=shuffle)
  
  def copy_weights(self, model):
    model.model.set_weights(self.model.get_weights())
  
  def save(self, filename):
    self.model.save(filename)
  
  def load(self, filename):
    self.model = tf.keras.models.load_model(filename)
"""

class Model(nn.Module):
  def __init__(self, input_shape, nb_actions):
    super().__init__()
    self.seq = nn.Sequential(
      nn.Linear(input_shape, 24),
      nn.ReLU(inplace=True),
      nn.Linear(24, 12),
      nn.ReLU(inplace=True),
      nn.Linear(12, nb_actions)
    )
  
  def forward(self, x):
    return self.seq(x)

class DenseNN:
  def __init__(self, input_shape, nb_actions):
    network = Model(input_shape, nb_actions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lossF = nn.MSELoss()
    lr = 0.001
    optim = torch.optim.Adam(network.parameters(), lr=lr)
    self.model = TorchWrapper(network, device, optim, lossF)
    print(self.model.get_parameters())
  
  def predict(self, X):
    return self.model.predict(X, num_workers=0)
  
  def fit(self, X, y, batch_size, shuffle):
    return self.model.fit(
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            verbose=False)
  
  def copy_weights(self, dense_nn):
    dense_nn.model.nn.load_state_dict(self.model.nn.state_dict())
  
  def save(self, filename):
    self.model.save(filename)
  
  def load(self, filename):
    self.model.load(filename)
