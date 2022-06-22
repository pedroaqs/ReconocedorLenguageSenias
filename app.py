from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk
import imutils
import mediapipe as mp
import sys, os
import numpy as np

class App:
    kernel_dilatacion = np.ones((5,5), np.uint8)
    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cont = 0
    dir_Entrenamiento = None
    dir_Prueba = None
    dir_Entrenamiento_1 = None
    dir_Prueba_1 = None
    write = False
    cls_manos = mp.solutions.hands
    manos = cls_manos.Hands()
    dibujo = mp.solutions.drawing_utils
    def __init__(self,ventana):
        self.ventana = ventana
        self.path =os.path.dirname(sys.argv[0])
        self.ventana.title("Capturar imagenes")
        self.ventana.iconbitmap("icono.ico")
        self.ventana.geometry("1300x600")
        self.ventana.resizable(False, False)
        self.canvas = Canvas(
            ventana,
            bg="#FFFFFF",
            height=600,
            width=1300,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        self.canvas.create_rectangle(0.0,0.0,300.0,600.0,fill="#3D7695",outline="")
        #Para ingresar la letra
        self.lbl_input = Label(self.canvas,text="Ingrese letra o número: ",bg="#3D7695")
        self.lbl_input.place(x=35.0,y=100,width=230.0,height=30.0)
        self.input = Entry(self.canvas, bd=0,bg="#FFFFFF",highlightthickness=0,justify="center")
        self.input.place(x=35.0,y=140.0,width=230.0,height=30.0)
        #Para escojer si seran imagenes de entrenamiento o prueba
        self.valor_rdb= IntVar()

        self.lbl_rdb = Label(self.canvas, text="Escoja una opcion: ", bg="#3D7695")
        self.lbl_rdb.place(x=35.0, y=190, width=230.0, height=30.0)
        self.rdb_entrenamiento = Radiobutton(self.canvas, text="Entrenamiento",bg="#3D7695", width=20, value=1, variable=self.valor_rdb)
        self.rdb_entrenamiento.place(x=35.0, y=220, width=230.0, height=30.0)
        self.rdb_prueba = Radiobutton(self.canvas, text="Prueba",bg="#3D7695", width=20, value=2, variable=self.valor_rdb)
        self.rdb_prueba.place(x=35.0, y=250, width=230.0, height=30.0)

        #Barra de progreso
        self.pgbar = ttk.Progressbar(self.canvas,style="TProgressbar", mode='indeterminate', length=400)
        self.pgbar.place(x=35.0, y=450, width=230.0, height=30.0)
        #boton

        self.btn_capturar = Button(self.canvas,text="Iniciar captura de imagenes",command=self.iniciarCaptura)
        self.btn_capturar.place(x=35.0, y=400, width=230.0, height=30.0)

        self.canvas.create_rectangle(300.0, 0.0, 1100.0, 600.0, fill="#FFFFFF", outline="")
        self.lbl_video =Label(self.canvas,text="No hay video")
        self.lbl_video.place(x=300.0, y=00, width=800.0, height=600.0)
        self.lbl_proccess1=Label(self.canvas,text = "Sin imagen")
        self.lbl_proccess1.place(x=1100.0, y=00, width=200.0, height=200.0)
        self.lbl_proccess2=Label(self.canvas, text="Sin imagen")
        self.lbl_proccess2.place(x=1100.0, y=200, width=200.0, height=200.0)
        self.lbl_proccess3=Label(self.canvas, text="Sin imagen")
        self.lbl_proccess3.place(x=1100.0, y=400, width=200.0, height=200.0)
        self.verVideo()

    def verVideo(self,):

        ret, frame = self.video.read()
        if ret == True:
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
                        #self.dibujo.draw_landmarks(frame, mano, self.cls_manos.HAND_CONNECTIONS)
                    if len(posiciones) != 0:
                        lista_x = []
                        lista_y = []
                        for item in posiciones:
                            lista_x.append(item[1])
                            lista_y.append(item[2])
                        x1, y1 = (min(lista_x) - 40), (min(lista_y) - 40)
                        x2, y2 = (max(lista_x) + 40), (max(lista_y) + 40)
                        dedos_reg = copia[y1:y2, x1:x2]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 3)
                    if dedos_reg.any():
                        dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                        img_p = dedos_reg
                        img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
                        img_p_1 = Image.fromarray(img_p)
                        img_1 = ImageTk.PhotoImage(image=img_p_1)
                        self.lbl_proccess1.configure(image=img_1)
                        self.lbl_proccess1.image = img_1
                        img_p_gris = cv2.cvtColor(img_p,cv2.COLOR_BGR2GRAY)
                        img_p_2 = Image.fromarray(img_p_gris)
                        img_2 = ImageTk.PhotoImage(image=img_p_2)
                        self.lbl_proccess2.configure(image=img_2)
                        self.lbl_proccess2.image = img_2
                        img_p_borde = cv2.Canny(img_p,200,200)
                        img_p_borde = cv2.dilate(img_p_borde,kernel=self.kernel_dilatacion,iterations=1)
                        img_p_borde = cv2.morphologyEx(img_p_borde, cv2.MORPH_OPEN, kernel = self.kernel_dilatacion)
                        img_p_3 = Image.fromarray(img_p_borde)
                        img_3 = ImageTk.PhotoImage(image=img_p_3)
                        self.lbl_proccess3.configure(image=img_3)
                        self.lbl_proccess3.image = img_3
                        if self.valor_rdb.get()==1 and self.write == True:
                            cv2.imwrite(self.dir_Entrenamiento + "/Mano_{}.jpg".format(self.cont), dedos_reg)
                            cv2.imwrite(self.dir_Entrenamiento_1 + "/Mano_{}.jpg".format(self.cont), img_p_borde)
                            self.cont = self.cont + 1
                        elif self.valor_rdb.get() == 2 and self.write == True:
                            cv2.imwrite(self.dir_Prueba + "/Mano_{}.jpg".format(self.cont), dedos_reg)
                            cv2.imwrite(self.dir_Prueba_1 + "/Mano_{}.jpg".format(self.cont), img_p_borde)
                            self.cont = self.cont + 1
                    if self.cont >= 500 and self.write == True:
                        self.valor_rdb.set(0)
                        self.pgbar.stop()
                        self.write = False
                        print("Terminado")
            frame = imutils.resize(frame, width=800)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            self.lbl_video.configure(image=img)
            self.lbl_video.image = img
            self.lbl_video.after(10, self.verVideo)
        else:
            self.lbl_video.image = ""
            self.video.release()

    def iniciarCaptura(self):
        self.pgbar.start()
        self.cont = 0
        self.write=True
        nombre = self.input.get()
        self.dir_Entrenamiento = self.path + '/Imagenes/Entrenamiento/' + nombre
        self.dir_Prueba = self.path + '/Imagenes/Pruebas/' + nombre
        self.dir_Entrenamiento_1 = self.path + '/Imagenes_proces/Entrenamiento/' + nombre
        self.dir_Prueba_1 = self.path + '/Imagenes_proces/Pruebas/' + nombre
        if self.valor_rdb.get() == 1 and self.write == True:
            if not os.path.exists(self.dir_Entrenamiento):
                print('Carpeta creada: ', self.dir_Entrenamiento)
                os.makedirs(self.dir_Entrenamiento)
                os.makedirs(self.dir_Entrenamiento_1)
        if self.valor_rdb.get() == 2 and self.write == True:
            if not os.path.exists(self.dir_Prueba):
                print('Carpeta creada: ', self.dir_Prueba)
                os.makedirs(self.dir_Prueba)
                os.makedirs(self.dir_Prueba_1)

if __name__ == "__main__":
    ventana=Tk()
    app=App(ventana)
    ventana.mainloop()
