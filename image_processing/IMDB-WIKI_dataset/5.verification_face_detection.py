'''
/***************************************************************************
 *                                                                         *
 *  File:        verification_face_detection.py                            *
 *  Copyright:   (c) 2020, Maria Frentescu                                 *
 *  Description: This script is used for double verification for face      *
 *               detection                                                 *
 **************************************************************************/
 '''
from random import sample
import os
from os.path import isfile,join
import cv2

face_cascade = cv2.CascadeClassifier('D:/licenta/interfata/haarcascade_frontalface_default.xml')
for i in range(18,51):
    prediction_path = "D:/SET_DATE/train/" + str(i)
    files = os.listdir(prediction_path)

    for img in files:
        dstPath = join(prediction_path, img)
        img = cv2.imread(dstPath)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        flag = 0
        for (x, y, w, h) in faces:
            flag = 1
        # flag=0 means no face detected
        # delete photos with no face
        if flag == 0:
            os.remove(dstPath)