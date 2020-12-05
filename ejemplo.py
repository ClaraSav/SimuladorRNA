

# Simple CNN para MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# ajusta el orden
from keras import backend as K
# K.set_image_dim_ordering('th')
# semillas de reproducibilidad
seed = 7
numpy.random.seed(seed)
# cargamos los datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reformas para [ejemplo][canales][ancho][alto]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normaliza entradas de 0-255 a 0-1
X_train = X_train / 255
X_test = X_test / 255
# codificacion en caliente para las salidas
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define el CNN model
def baseline_model():
  # crea el modelo
  model = Sequential()
  model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile el modelo
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
 
# construimos el modelo
model = baseline_model()
# ajusta el modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluacion del modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
