import torch
import torch.nn as nn
from utils import RANDOM_SEED
from torch_model_wrapper import TorchWrapper
torch.manual_seed(RANDOM_SEED)

class Model(nn.Module):
  def __init__(self, input_shape, nb_actions):
    super().__init__()
    self.seq = nn.Sequential(
      nn.Linear(input_shape, 24),
      nn.ReLU(inplace=True),
      nn.Linear(24, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 16),
      nn.ReLU(inplace=True),
      nn.Linear(16, nb_actions)
    )
  
  def forward(self, x):
    return self.seq(x)

class DenseNN:
  def __init__(self, input_shape, nb_actions):
    network = Model(input_shape, nb_actions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    lossF = nn.MSELoss()
    lr = 0.001
    optim = torch.optim.Adam(network.parameters(), lr=lr)
    self.model = TorchWrapper(network, device, optim, lossF)
  
  def predict(self, X):
    return self.model.predict(X, num_workers=0)
  
  def fit(self, X, y, batch_size, shuffle, tensorboard_writer, epochs=1):
    return self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            tensorboard_writer=tensorboard_writer,
            verbose=False)
  
  def copy_weights(self, dense_nn):
    dense_nn.model.nn.load_state_dict(self.model.nn.state_dict())
  
  def save(self, filename):
    self.model.save(filename)
  
  def load(self, filename):
    self.model.load(filename)