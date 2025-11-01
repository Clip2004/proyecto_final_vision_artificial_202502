import cv2
import numpy as np
from glob import glob
import os
import xlsxwriter
from math import pi

## DIR NAMES
inputfolder = 'tarea_banda/banda_modelos/clases_neural_network_COPY'
subfolders = ['clase_argollas', 'clase_ochos', 'clase_Tensores', 'clase_Z_1', 'clase_Z_2']
labels = {name: idx for idx, name in enumerate(subfolders)}
## DIR NAMES


row = 0
col = 1
workbook = xlsxwriter.Workbook('tarea_banda/banda_modelos/charact_argollas.xlsx')
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
        for imgpath in glob(inputfolder + '/' + subfolders[j] + f'/*.png'):
            img = cv2.imread(imgpath, 1)
            img_contour_drawn = img.copy()
            imgcopy = img.copy()
            # # resize img
            newsize = res_using_ar(imgcopy, 0.15)
            imgcopy = cv2.resize(imgcopy, newsize)
            img_bin = cv2.inRange(imgcopy, (0,0,19), (255,255,255))
            
            # img_bin = preprocess_digit(img, output_size=(5, 5))


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
                                        # x12,
                                        # x13,
                                        # x14,
                                        ], 
                                        dtype=np.float32)

            for pattern in vector_patterns:
                worksheet.write(row, 0, labels[subfolders[j]])
                worksheet.write(row, col, pattern)
                col += 1
            col = 1 # reset columns
            row += 1
                

            # dst = cv2.resize(roibin, (320, 480))
            cv2.imshow('ROI', img_bin)
            cv2.imshow('img', img_contour_drawn)
            cv2.waitKey(5)

    cv2.destroyAllWindows()
    workbook.close()
    print('FINISHED')

extract_chars()
