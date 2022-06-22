from tkinter import *
import cv2
import numpy as np
import mediapipe as mp
from keras_preprocessing.image import  load_img,img_to_array
from keras.models import load_model
import sys,os
import imutils
from PIL import Image
from PIL import ImageTk

class App:

    modelo = os.path.dirname(sys.argv[0]) + "/Modelo.h5"
    peso = os.path.dirname(sys.argv[0]) + "/pesos.h5"

    dir_img = os.listdir(os.path.dirname(sys.argv[0]) + "/Imagenes_proces/Pruebas")
    print(dir_img)
    kernel_dilatacion = np.ones((5, 5), np.uint8)

    cnn = load_model(modelo)
    cnn.load_weights(peso)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    clase_manos = mp.solutions.hands

    manos = clase_manos.Hands()
    dibujo = mp.solutions.drawing_utils
    def __init__(self, ventana):
        self.ventana = ventana
        self.path = os.path.dirname(sys.argv[0])
        self.ventana.title("Capturar imagenes")
        self.ventana.iconbitmap("icono.ico")
        self.ventana.geometry("1100x600")
        self.ventana.resizable(False, False)
        self.canvas = Canvas(
            ventana,
            bg="#FFFFFF",
            height=600,
            width=1100,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        self.canvas.create_rectangle(0.0, 0.0, 300.0, 600.0, fill="#3D7695", outline="")
        self.canvas.create_rectangle(300.0, 0.0, 1100.0, 600.0, fill="#FFFFFF", outline="")
        self.lbl_rpt = Label(self.canvas,text="RESULTADO",font=("Arial", 30,"bold"),bg="#3D7695",fg="#ffffff",)
        self.lbl_rpt.place(x=0.0,y=100.0,width=300.0, height=200.0)
        self.lbl_video = Label(self.canvas, text="No hay video")
        self.lbl_video.place(x=300.0, y=00, width=800.0, height=600.0)
        self.lbl_imagen_process = Label(self.canvas, text="No hay video")
        self.lbl_imagen_process.place(x=0.0, y=400.0, width=300.0, height=200.0)
        self.verVideo()

    def verVideo(self, ):
        ret, frame = self.cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia = frame.copy()
        resultado = self.manos.process(color)
        posiciones = []
        if resultado.multi_hand_landmarks:
            for mano in resultado.multi_hand_landmarks:
                for id, lm in enumerate(mano.landmark):
                    alto, ancho, c = frame.shape
                    cordx, cordy = int(lm.x * ancho), int(lm.y * alto)
                    posiciones.append([id, cordx, cordy])
                    #self.dibujo.draw_landmarks(frame, mano, self.clase_manos.HAND_CONNECTIONS)
                if len(posiciones) != 0:
                    lista_x = []
                    lista_y = []
                    for item in posiciones:
                        lista_x.append(item[1])
                        lista_y.append(item[2])
                    x1, y1 = (min(lista_x) - 40), (min(lista_y) - 40)
                    x2, y2 = (max(lista_x) + 40), (max(lista_y) + 40)
                    dedos_reg = copia[y1:y2, x1:x2]
                    if dedos_reg.any():
                        #print("D", dedos_reg)
                        dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                        img_p_borde = cv2.Canny(dedos_reg, 200, 200)
                        img_p_borde = cv2.dilate(img_p_borde, kernel=self.kernel_dilatacion, iterations=1)
                        img_p_borde = cv2.morphologyEx(img_p_borde, cv2.MORPH_OPEN, kernel=self.kernel_dilatacion)
                        img_p_borde = cv2.cvtColor(img_p_borde,cv2.COLOR_GRAY2RGB)
                        img_p_borde = cv2.resize(img_p_borde, (200, 200), interpolation=cv2.INTER_CUBIC)

                        if img_p_borde.any() :
                            x = img_to_array(img_p_borde)
                            x = np.expand_dims(x, axis=0)
                            vector = self.cnn.predict(x)
                            img_p_borde = cv2.resize(img_p_borde, (300, 200), interpolation=cv2.INTER_CUBIC)
                            img_p_3 = Image.fromarray(img_p_borde)
                            img_3 = ImageTk.PhotoImage(image=img_p_3)
                            self.lbl_imagen_process.configure(image=img_3)
                            self.lbl_imagen_process.image = img_3
                            resultado = np.transpose(vector[0])
                            respuesta = np.argmax(resultado)
                            print(resultado)
                            #if resultado[respuesta]==1:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 3)
                            self.lbl_rpt.configure(text=self.dir_img[respuesta],font=("Arial", 80,"bold"))
                            self.lbl_rpt.text=self.dir_img[respuesta]

        frame = imutils.resize(frame, width=800)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        self.lbl_video.configure(image=img)
        self.lbl_video.image = img
        self.lbl_video.after(10, self.verVideo)

if __name__ == "__main__":
    ventana = Tk()
    app = App(ventana)
    ventana.mainloop()