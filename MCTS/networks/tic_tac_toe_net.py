import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_network(input_shape, nb_actions):
  inputs = layers.Input(input_shape)

  x_old = layers.Conv2D(64, (1, 1))(inputs)
  x_old = layers.ReLU()(x_old)
  
  x = layers.Conv2D(64, (1, 1))(x_old)
  x = layers.ReLU()(x)
  x = layers.Conv2D(64, (1, 1))(x)
  x = layers.ReLU()(x)
  x_old = layers.Add()([x, x_old])

  x = layers.Conv2D(64, (1, 1))(x_old)
  x = layers.ReLU()(x)
  x = layers.Conv2D(64, (1, 1))(x)
  x = layers.ReLU()(x)
  x_old = layers.Add()([x, x_old])

  x = layers.Conv2D(64, (1, 1))(x_old)
  x = layers.ReLU()(x)
  x = layers.Conv2D(64, (1, 1))(x)
  x = layers.ReLU()(x)
  x_old = layers.Add()([x, x_old])

  outputs = layers.Flatten()(x_old)

  policy = layers.Dense(nb_actions, activation="softmax", name="policy")(outputs)
  value = layers.Dense(1, activation="tanh", name="value")(outputs)
  return tf.keras.Model(inputs=inputs, outputs=[policy, value])