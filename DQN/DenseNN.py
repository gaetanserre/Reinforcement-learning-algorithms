import tensorflow as tf
from utils import RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)

class DenseNN:
  def __init__(self, input_shape, nb_action):
    learning_rate = 0.001
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Dense(24, input_shape=(input_shape,), activation="relu"))
    self.model.add(tf.keras.layers.Dense(12, input_shape=(input_shape,), activation="relu"))
    self.model.add(tf.keras.layers.Dense(nb_action, activation="linear"))
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