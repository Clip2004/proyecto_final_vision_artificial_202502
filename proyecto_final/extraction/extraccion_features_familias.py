import cv2
import numpy as np
from glob import glob
import xlsxwriter
from math import pi

# Directorios
inputfolder = 'proyecto_final/contornos_database'
subfolders = ['argollas', 'tensores', 'zetas']
labels = {name: idx for idx, name in enumerate(subfolders)}

row = 0
col = 1

workbook = xlsxwriter.Workbook('proyecto_final/extraction/charact_familias.xlsx')
worksheet = workbook.add_worksheet('patterns')

def extract_chars():
    global row, col

    for j in range(len(subfolders)):
        folder_path = f"{inputfolder}/{subfolders[j]}/*.jpg"
        print("Buscando imágenes en:", folder_path)

        for imgpath in glob(folder_path):

            print("Procesando:", imgpath)

            # Leer en gris
            img_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print("Error leyendo:", imgpath)
                continue

            print(img_gray)

          

            # cv2.imshow("Gris", img_gray)
            # cv2.waitKey(1)
            
            
            # Detectar contornos para ROI
            conts, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                
            # Si no hay contornos → saltar imagen
            if len(conts) == 0:
                continue

            # Escoger contorno más grande
            maxcont = max(conts, key=cv2.contourArea)

            # Filtrar ruido
            if cv2.contourArea(maxcont) < 50:
                continue
                        

            # ROI y normalización
            x, y, w, h = cv2.boundingRect(maxcont)
            roi = img_gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_NEAREST)

            # -------- Features --------

            # Contorno principal
            x1_area = cv2.contourArea(maxcont)
            x2_area = cv2.contourArea(cv2.convexHull(maxcont))
            x3_perimeter = cv2.arcLength(maxcont, True)
            x4_circle = x1_area*4*pi/(x3_perimeter**2)

            # Hu Moments
            M = cv2.moments(maxcont)
            Hu = cv2.HuMoments(M)
            x5,x6,x7,x8,x9,x10,x11 = [Hu[i][0] for i in range(7)]

            # Densidad 1/4 altura
            H = roi_resized.shape[0]
            x12 = np.sum(roi_resized[H//4, :] / 255) / (x1_area + 1e-6)

            # Aspect ratio bounding box original
            x13 = w / (h + 1e-6)
            x14 = h / (w + 1e-6)

            # Contornos internos en ROI
            conts_all, hierarchy2 = cv2.findContours(roi_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   

            # Simetría
            x15 = np.mean(roi_resized == np.flipud(roi_resized))
            x16 = np.mean(roi_resized == np.fliplr(roi_resized))

            # Área relativa de huecos internos
            num_interior = 0
            area_ratio_interior = 0.0
            
            if hierarchy2 is not None:
                hierarchy2 = hierarchy2[0]  # aplanar
                for i in range(len(hierarchy2)):
                    if hierarchy2[i][3] != -1:  # tiene padre → contorno interior
                        num_interior += 1
                        area_ratio_interior += cv2.contourArea(conts_all[i]) / (x1_area + 1e-6)
            x17 = area_ratio_interior / num_interior if num_interior > 0 else 0

            # Vector final de features
            vector_patterns = np.array([
                x1_area, 
                x2_area,
                x3_perimeter, 
                x4_circle,
                x5, 
                x6, 
                x7, 
                x8, 
                x9, 
                x10, 
                x11,
                x12, 
                x13, 
                x14, 
                x15, 
                x16, 
                x17
            ], dtype=np.float32)

            # Guardar fila en Excel
            for pattern in vector_patterns:
                worksheet.write(row, 0, labels[subfolders[j]])
                worksheet.write(row, col, pattern)
                col += 1

            col = 1
            row += 1

            cv2.imshow("ROI", roi_resized)
            cv2.imshow("Original", img_gray)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    workbook.close()
    print("Extracción finalizada.")

extract_chars()
