'''
/***************************************************************************
 *                                                                         *
 *  File:        sort_ages.py                                              *
 *  Copyright:   (c) 2020, Maria Frentescu                                 *
 *  Description: This script is used to sort all images from a folder to   *
 *               100 folders using the value of age from image name.       *
 *                                                                         *
 **************************************************************************/
'''
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join

path='D:/licenta/set_date_3/UTKFace/'
# Making the directory structure
for i in range(1,101):
    dstPath = 'D:/licenta/set_date_3/sorted/train/' + str(i)
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
        
files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:

    filename_w_ext = os.path.basename(image)
    filename, file_extension = os.path.splitext(filename_w_ext)
    year=filename.split('_')[0]
    img = cv2.imread(os.path.join(path,image))
    img = cv2.resize(img, (128,128))
    cv2.imwrite('D:/licenta/set_date_3/sorted/train/' + str(year) + '/' + filename + '.jpg', img)








