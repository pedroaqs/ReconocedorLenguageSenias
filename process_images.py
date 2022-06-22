import cv2
import os
import numpy as np

letra = "/Pruebas/Z"

path_images = "D:/Universidad/10 ciclo/Sistemas Inteligentes/code/Proyecto-SI/Imagenes_falla" + letra
path_save = "D:/Universidad/10 ciclo/Sistemas Inteligentes/code/Proyecto-SI/for_train" + letra


images_name = os.listdir(path_images)
kernel_dilatacion = np.ones((5,5), np.uint8)

for image in images_name:
    image_p = path_images+"/"+image
    print(image_p)

    img_p = cv2.imread(image_p)
    img_p_borde = cv2.Canny(img_p, 200, 200)
    img_p_borde = cv2.dilate(img_p_borde, kernel=kernel_dilatacion, iterations=1)
    img_p_borde = cv2.morphologyEx(img_p_borde, cv2.MORPH_OPEN, kernel=kernel_dilatacion)
    cv2.imwrite(path_save + "/"+image, img_p_borde)

cv2.destroyAllWindows()