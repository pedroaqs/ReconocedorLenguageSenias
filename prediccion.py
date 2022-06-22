import cv2
import mediapipe as mp
import sys,os
import numpy as np
from keras_preprocessing.image import  load_img,img_to_array
from keras.models import load_model


modelo=os.path.dirname(sys.argv[0]) + "/Modelo.h5"
peso=os.path.dirname(sys.argv[0]) + "/pesos.h5"

dir_img=os.listdir(os.path.dirname(sys.argv[0]) + "/Imagenes_falla/Pruebas")


cnn = load_model(modelo)
cnn.load_weights(peso)

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

clase_manos = mp.solutions.hands

manos = clase_manos.Hands()
dibujo =mp.solutions.drawing_utils

while True:
    ret,frame = cap.read()
    color = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado= manos.process(color)
    posiciones = []
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                cordx, cordy = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, cordx, cordy])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                lista_x = []
                lista_y = []
                for item in posiciones:
                    lista_x.append(item[1])
                    lista_y.append(item[2])
                x1, y1 = (min(lista_x) - 100), (min(lista_y) - 100)
                x2, y2 = (max(lista_x) + 100), (max(lista_y) + 100)
                dedos_reg = copia[y1:y2, x1:x2]
                if dedos_reg.any():
                    dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                    x= img_to_array(dedos_reg)
                    x= np.expand_dims(x,axis=0)
                    vector = cnn.predict(x)
                    print(cnn.predict(x))
                    resultado = np.transpose(vector[0])
                    respuesta = np.argmax(resultado)
                    if respuesta == 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 3)
                        cv2.putText(frame, dir_img[0], (x1, y1-10), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                    elif respuesta == 1:
                        print(vector, resultado)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 3)
                        cv2.putText(frame, dir_img[1], (x1, y1 - 10), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Prediccion", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()