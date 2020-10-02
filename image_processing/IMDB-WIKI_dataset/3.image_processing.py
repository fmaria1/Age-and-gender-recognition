'''
/***************************************************************************
 *                                                                         *
 *  File:        face_detection.py                                         *
 *  Copyright:   (c) 2020, Maria Frentescu                                 *
 *  Description: This script provide face detection for raw images and     *
 *               save new version only with face picture                   *
 *                                                                         *
 **************************************************************************/
 '''
import cv2
import os,glob
from PIL import ImageTk
from PIL import Image
from os import listdir,makedirs
from os.path import isfile,join

for i in range(18,51):
    path = 'D:/licenta/procesare_imagini/set_20_50/train/' + str(i) # Source Folder
    dstpath = 'D:/licenta/procesare_imagini/croped/train/' + str(i) # Destination Folder
    try:
        makedirs(dstpath)
    except:
        print ("Directory already exist, images will be written in same folder")
    files = [f for f in listdir(path) if isfile(join(path,f))]
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for image in files:
        try:
        # Detect Faces
            img = cv2.imread(os.path.join(path,image))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            flag = 0
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                flag = 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
                # if one face detected, move to next image
                # in case for more faces in photo
                if flag == 1:
                    break

            img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img3)

            width, height = im.size
            left = x
            top = y
            right = x + w
            bottom = y + h

            # Cropped image of above dimension
            im1 = im.crop((left, top, right, bottom))
            # resize all images to 128x128 pixels
            im1 = im1.resize((128, 128), Image.ANTIALIAS)
            image1 = ImageTk.PhotoImage(image=im1)

            dstPath = join(dstpath,image1)
            cv2.imwrite(dstPath,im1)
        except:
            print ("{} is not converted".format(image))
    # convert to grayscale if needed
    gray_convert = 0
    if gray_convert == 1:
        for fil in glob.glob("*.jpg"):
            try:
                image = cv2.imread(fil)            
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                cv2.imwrite(os.path.join(dstpath,fil),gray_image)
            except:
                print('{} is not converted').format(image)