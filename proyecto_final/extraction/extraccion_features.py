import cv2
import numpy as np
from glob import glob
import os
import xlsxwriter
from math import pi

## DIR NAMES
inputfolder = 'proyecto_final/contornos_database'

subfolders = ['argollas', 'tensores', 'zetas']
labels = {name: idx for idx, name in enumerate(subfolders)}
## DIR NAMES


row = 0
col = 1
workbook = xlsxwriter.Workbook('proyecto_final/extraction/charact_familias.xlsx')
worksheet = workbook.add_worksheet('patterns')


# ar == 'aspect ratio'
def res_using_ar(img, factor):
    size = img.shape
    newsize = list(map(lambda x:int(x*factor), size))[0:2]
    newsize.reverse()
    return newsize


def preprocess_digit(img, output_size=(128, 128)):
    # Convert to grayscale and binary (if not done already)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.inRange(img, (0,0,19), (255,255,255))

    # Find bounding box of the digit
    x, y, w, h = cv2.boundingRect(binary)
    digit_roi = binary[y:y+h, x:x+w]

    # Resize keeping aspect ratio
    aspect = w / h
    if aspect > 1:
        new_w = output_size[0]
        new_h = int(output_size[0] / aspect)
    else:
        new_h = output_size[1]
        new_w = int(output_size[1] * aspect)

    resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to make final size consistent
    top = (output_size[1] - new_h) // 2
    bottom = output_size[1] - new_h - top
    left = (output_size[0] - new_w) // 2
    right = output_size[0] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return padded


def extract_chars():
    global row, col
    for j in range(len(subfolders)):
        print("Buscando imágenes en:", inputfolder + '/' + subfolders[j] + f'/*.jpg')

        for imgpath in glob(inputfolder + '/' + subfolders[j] + f'/*.jpg'):

            print("Leyendo:", imgpath)

            # ✅ leer directamente en escala de grises
            img_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

            if img_gray is None:
                print(f"Error leyendo: {imgpath}")
                continue
            
            # Hacemos copia para dibujar y procesar
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # solo para dibujar contornos
            img_contour_drawn = img.copy()
            
            # ✅ Resize (manteniendo la lógica original)
            newsize = res_using_ar(img_gray, 0.15)
            imgcopy = cv2.resize(img_gray, newsize)

            # ✅ Como ya viene binarizada, usamos la imagen tal cual
            img_bin = imgcopy  


            conts = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

            if len(conts) > 0:
                maxcont = max(conts, key=cv2.contourArea)
            else:
                continue # keep going with the for loop

            cv2.drawContours(img_contour_drawn, maxcont, -1, (0, 255, 0), 3)

            ##! BIG NOTE
            # from x1 to x12 are inputs to the neural network that is going to predict what number is on the image
            ##! BIG NOTE

            x1_area = cv2.contourArea(maxcont)
            x2_area = cv2.contourArea(cv2.convexHull(maxcont))
            x3_perimeter = cv2.arcLength(maxcont, True)
            x4_circle = x1_area*4*pi/(x3_perimeter**2)

            M = cv2.moments(maxcont)

            Hu = cv2.HuMoments(M)

            x5  = Hu[0][0]
            x6  = Hu[1][0]
            x7  = Hu[2][0]
            x8  = Hu[3][0]
            x9  = Hu[4][0]
            x10 = Hu[5][0]
            x11 = Hu[6][0]

            # extract number of pixels on the first row on 'roibin'
            # x12 = np.sum(roibin[5, 10])/x1_area

            w, h = img_bin.shape
            x12 = np.sum(img_bin[round(h/4), :]/255)/x1_area
            
            x13 = w/x1_area
            x14 = h/x1_area

            # num contornos

            conts_all = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            x15 = len(conts_all)   

            # jerarquia
            hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[1]
            x16 = np.sum(hierarchy[:,:,3] != -1)

            # scores de simetria horizontal y vertical
            x17 = np.mean(img_bin == np.flipud(img_bin))
            x18   = np.mean(img_bin == np.fliplr(img_bin))

            # zone_features = []
            # cell_size = 5

            # for y in range(0, img_bin.shape[0], cell_size):
            #     for x in range(0, img_bin.shape[1], cell_size):
            #         roi_zone = img_bin[y:y+cell_size, x:x+cell_size]
            #         density = cv2.countNonZero(roi_zone) / (cell_size * cell_size)
            #         zone_features.append(density)
            # store the neural network inputs in a numpy array
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
                                        x17,
                                        x18,
                                        
                                        ], 
                                        dtype=np.float32)
            
            #zone_features = np.array(zone_features, dtype=np.float32)

            # vector_patterns = np.concatenate([vector_patterns, zone_features])

            for pattern in vector_patterns:
                worksheet.write(row, 0, labels[subfolders[j]])
                worksheet.write(row, col, pattern)
                col += 1
            col = 1 # reset columns
            row += 1
                

            # dst = cv2.resize(roibin, (320, 480))
            cv2.imshow('ROI', img_bin)
            cv2.imshow('img', img_contour_drawn)
            key = cv2.waitKey(0) & 0xFF  # Espera indefinidamente hasta que se presione una tecla
            # if key == ord('q'):  # Si se presiona 'q'
            #     break  # Sale del bucle

    cv2.destroyAllWindows()
    workbook.close()
    print('FINISHED')

extract_chars()
