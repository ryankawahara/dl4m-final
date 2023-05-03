from tensorflow import keras
from keras.optimizers import Adam



def build_dense_model(input_shape):
  """
  Build and compile a dense neural network model with a
  sigmoid output layer for binary classification.

  Parameters
  ----------
  input_shape : tuple
      The shape of the input tensor, excluding the batch size.

  Returns
  -------
  tf.keras.Model
      The compiled dense neural network model.

  Notes
  -----
  The dense neural network model consists of a flatten layer, a dense layer with 256 units, a dropout layer with 0.5
  dropout rate, and a sigmoid output layer. The binary cross-entropy loss function is used for training, and the Adam
  optimizer is used for optimization. The model is evaluated based on the accuracy metric during training.

  Examples
  --------
  >>> model = build_dense_model(input_shape=(28, 28, 1))
  >>> model.summary()
  """
  inputs = keras.Input(shape=input_shape)
  x = keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu")(inputs)

  # x = keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu")(inputs)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dropout(.75)(x)
  # outputs = keras.layers.Dense(10, activation="softmax")(x)

  outputs = keras.layers.Dense(10, activation="sigmoid")(x)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(loss="binary_focal_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
  return model



# inputs = keras.Input(shape=input_shape)
#   x = keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu")(inputs)
#   x = keras.layers.Dense(256, activation='relu')(x)
#   x = keras.layers.MaxPooling2D(pool_size=2)(x)
#   x = keras.layers.Flatten()(x)
#   x = keras.layers.Dropout(.8)(x)
#   outputs = keras.layers.Dense(10, activation="sigmoid")(x)
#   model = keras.Model(inputs=inputs, outputs=outputs)
#
#   model.compile(loss="binary_crossentropy",
#             optimizer="adam",
#             metrics=["accuracy"])
#   return model

def load_conv_model(weights='imagenet', include_top=False, input_shape=(180, 180, 3)):
  """
  Loads the VGG16 convolutional base model.

  Parameters
  ----------
  weights : str or None, optional
      Specifies which weights to load for the model. It can be either 'imagenet'
      (pre-training on ImageNet) or None (random initialization). The default
      value is 'imagenet'.
  include_top : bool, optional
      Whether to include the fully connected layers at the top of the network.
      The default value is False, which means the last fully connected layers
      are excluded.
  input_shape : tuple of int, optional
      The shape of the input tensor to the model. The default shape is (180, 180, 3).

  Returns
  -------
  A Keras model object.
  """
  # Hint: use keras.applications

  vgg_model = keras.applications.VGG16(
    weights=weights,
    include_top=include_top,
    input_shape=input_shape
    )

  return vgg_model


def build_baseline(input_shape):
  """
  Parameters
  ----------
  input_shape : tuple of int
      The shape of the input tensor, e.g. (height, width, channels).

  Returns
  -------
  model : keras.Model
      The compiled baseline convolutional neural network model.

  Notes
  -----
  The model architecture consists of four convolutional blocks consisting
  of one convolutional and one max pooling layer:
  - Block 1: 32 filters, kernel size 3, activation function ReLU, pool size 2.
  - Block 2: 64 filters, kernel size 3, activation function ReLU, pool size 2.
  - Block 3: 128 filters, kernel size 3, activation function ReLU, pool size 2.
  - Block 4: 256 filters, kernel size 3, activation function ReLU, pool size 2.

  The last block is followed by a convolutional network

  output of the fourth convolutional block is followed by a convolkutional network
  with 256 filters, kernel size 3, activation function ReLU. The output of this conv layer is
  then flattened and connected to a dense layer with one neuron,
  which is activated by the sigmoid function. The model is compiled with binary crossentropy loss,
  Adam optimizer, and accuracy metric.
  """
  inputs = keras.Input(shape=input_shape)
  x = keras.layers.Dense(128, activation='relu')(inputs)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dropout(.8)(x)
  outputs = keras.layers.Dense(10, activation="sigmoid")(x)
  model = keras.Model(inputs=inputs, outputs=outputs)

  learning_rate = 0.0001  # initial learning rate
  optimizer = Adam(lr=learning_rate / 10)

  model.compile(loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
  return model

def build_reg_model(input_shape):
  """
  Parameters
  ----------
  input_shape : tuple of int
      The shape of the input tensor, e.g. (height, width, channels).

  Returns
  -------
  model : keras.Model
      The compiled baseline convolutional neural network model.

  Notes
  -----
  The model architecture is identical to the baseline but it has a
  Dropout layer with 0.5 dropout between the dense and rhe flatten layers.

  """
  inputs = keras.Input(shape=input_shape)
  x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)
  x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)
  x = keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)
  x = keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dropout(.5)(x)
  outputs = keras.layers.Dense(10, activation="sigmoid")(x)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
  return model


