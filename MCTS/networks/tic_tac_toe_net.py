import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_network(input_shape, nb_actions):
  filters = 256
  inputs = tf.keras.Input(input_shape)
  x = layers.Conv2D(filters, (1, 1))(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  for _ in range(6):
    x_old = x
    x = layers.Conv2D(filters, (1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1,1))(x)
    x = layers.BatchNormalization()(x)

    # SE
    x_se = layers.GlobalAveragePooling2D()(x)
    x_se = layers.Dense(filters//16, activation="relu")(x_se)
    x_se = layers.Dense(filters, activation="sigmoid")(x_se)
    x = layers.Multiply()([x, x_se])

    x = layers.Add()([x, x_old])
    x = layers.ReLU()(x)

  x = layers.Conv2D(filters, (1, 1))(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  x = layers.Flatten()(x)

  policy = layers.Dense(nb_actions, activation="softmax", name="policy")(x)
  value = layers.Dense(1, activation="tanh", name="value")(x)
  model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
  model._name = "TicTacToe_network"
  losses = {"policy": "categorical_crossentropy", "value": "mean_squared_error"}
  metrics = {"policy": "accuracy", "value": "mean_squared_error"}
  model.compile(loss=losses, optimizer="adam", metrics=metrics)
  return model