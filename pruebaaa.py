import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import cv2
datos_entrenamiento=os.path.dirname(sys.argv[0]) + "/A"
#datos_prueba = os.path.dirname(sys.argv[0]) + "/Imagenes_falla/Pruebas/A"

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

# generate batch of images
train_generator = datagen.flow_from_directory(
    datos_entrenamiento,
    target_size=(200, 200),
    batch_size=1,
    class_mode="categorical"
)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))

for i in range(4):

    # convert to unsigned integers for plotting
    image = next(train_generator)[0].astype('uint8')
    strg = "Imagen"+str(i)

    # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
    image = np.squeeze(image)
    cv2.imshow(strg, image)
    cv2.waitKey(0)
    # plot raw pixel data
    ax[i].imshow(image)
    ax[i].axis('off')