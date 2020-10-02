'''
/***************************************************************************
 *                                                                         *
 *  File:        crop_margins.py                                           *
 *  Copyright:   (c) 2020, Maria Frentescu                                 *
 *  Description: This script is used to crop 40% from 128x128 pixels image *
 *                                                                         *
 **************************************************************************/
 '''
import numpy as np
import cv2
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join

for i in range(0,2):
    path = 'D:/licenta/SET_DATE_FINAL/GENDER/test/' + str(i) # Source Folder
    dstpath = 'D:/licenta/SET_DATE_FINAL/CROPED_GENDER_FINAL/test/' + str(i) # Destination Folder
    try:
        makedirs(dstpath)
    except:
        print ("Directory already exist, images will be written in same folder")
    files = [f for f in listdir(path) if isfile(join(path,f))]
    for image in files:
        try:
            #crop 40% margins from images
            img = cv2.imread(os.path.join(path,image))
            y=26
            x=26
            h=77
            w=77
            crop = img[y:y+h, x:x+w]
            dstPath = join(dstpath,image)
            cv2.imwrite(dstPath,crop)
        except:
            print ("{} is not converted".format(image))
    


