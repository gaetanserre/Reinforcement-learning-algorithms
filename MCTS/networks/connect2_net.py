import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_network(input_shape, nb_actions):
  inputs = layers.Input(input_shape)
  outputs = layers.Dense(32)(inputs)

  policy = layers.Dense(nb_actions, activation="softmax", name="policy")(outputs)
  value = layers.Dense(1, activation="tanh", name="value")(outputs)
  model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
  model._name = "Connect2 network"
  return model