from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import  adam_v2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout,Flatten,Dense,Activation
from tensorflow.python.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.python.keras import  backend as k
import tensorflow
import sys, os
import time
k.clear_session()
print("Inicio: ",time.strftime("%c"))
datos_entrenamiento=os.path.dirname(sys.argv[0]) + "/Imagenes_proces/Entrenamiento"
datos_prueba = os.path.dirname(sys.argv[0]) + "/Imagenes_proces/Pruebas"


#Parametros
iteraciones = 2
altura,longitud =200,200
batch_size = 1
pasos = 1000/1
pasos_validacion = 1000/1
filtros_conv1=32
filtros_conv2=64
filtros_conv3=128
filtros_conv4=256
tam_filtro1= (5, 5)
tam_filtro2 =(5, 5)
tam_filtro3 =(3, 3)
tam_filtro4 =(3, 3)
tam_pool= (2, 2)
clases = 26
lr = 0.0005

preprocesamiento_entrenamiento = ImageDataGenerator(
    #rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

preprocesamiento_pruebas = ImageDataGenerator(
    #rescale=1./255
)

imagen_entreno = preprocesamiento_entrenamiento.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode="categorical"
)

imagen_pruebas= preprocesamiento_pruebas.flow_from_directory(
    datos_prueba,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical"
)

cnn=Sequential()
#primer filtro
cnn.add(Convolution2D(filtros_conv1,tam_filtro1,padding="same",input_shape=(altura,longitud,3),activation="relu"))
cnn.add(MaxPooling2D(pool_size=tam_pool))
#segundo filtro
cnn.add(Convolution2D(filtros_conv2,tam_filtro2,padding="same",activation="relu"))
cnn.add(MaxPooling2D(pool_size=tam_pool))
#tercer filtro
cnn.add(Convolution2D(filtros_conv3,tam_filtro3,padding="same",activation="relu"))
cnn.add(MaxPooling2D(pool_size=tam_pool))
#cuarto filtro
cnn.add(Convolution2D(filtros_conv4,tam_filtro4,padding="same",activation="relu"))
cnn.add(MaxPooling2D(pool_size=tam_pool))


cnn.add(Flatten())

#por cada clase son 128 neuronas

cnn.add(Dense(3328,activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation="softmax"))

optimizar = adam_v2.Adam(learning_rate=lr)


cnn.compile(loss="categorical_crossentropy", optimizer=optimizar, metrics=["accuracy"])

cnn.fit(imagen_entreno,steps_per_epoch=pasos,epochs=iteraciones,validation_data=imagen_pruebas,validation_steps=pasos_validacion)


cnn.save("/A/Modelo.h5")
cnn.save_weights("/A/pesos.h5")
print("Fin: ",time.strftime("%c"))

