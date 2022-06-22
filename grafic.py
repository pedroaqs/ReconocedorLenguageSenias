import matplotlib.pyplot as plt
from keras.models import load_model
import sys,os

modelo = os.path.dirname(sys.argv[0]) + "/Modelo.h5"
peso = os.path.dirname(sys.argv[0]) + "/pesos.h5"

cnn = load_model(modelo)
cnn.load_weights(peso)

plt.plot(cnn.history['accuracy'], label='accuracy')
plt.plot(cnn.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
