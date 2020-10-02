'''
/***************************************************************************
 *                                                                         *
 *  File:        proc_imdb-wiki_files.py                                   *
 *  Copyright:   (c) 2019, Maria Frentescu                                 *
 *  Description: Sorting all images in folders using age information       *
 *                                                                         *
 **************************************************************************/
 '''
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split

# load file with age information
meta = pd.read_csv('meta.csv')

# delete the column with gender info
meta = meta.drop(['gender'], axis=1)

# filter the dataset
meta = meta[meta['age'] >= 0]
meta = meta[meta['age'] <= 101]
meta = meta.values

# sorting sataset in train and test 
D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)

# make directory stucture
# create train dataset
counter = 0
for image in D_train:
    img = cv2.imread(image[1], 1)
    img = cv2.resize(img, (128, 128))
    nume = image[1][13:]
    if nume[0].islower() == True:
        found = nume.partition("_")[2].partition("-")[0]
        born_year = found.partition("_")[2].partition(" ")[0]
        found1 = nume[::-1].partition("gpj.")[2].partition("_")[0]
        photo_year = found1[::-1]
        agen = int(photo_year) - int(born_year)
    else:
        born_year = nume.partition("_")[2].partition("-")[0]
        found1 = nume[::-1].partition("gpj.")[2].partition("_")[0]
        photo_year = found1[::-1]
        agen = int(photo_year) - int(born_year)
    cv2.imwrite('dataset1/age/train/' + str(agen) + '/' + str(counter) + '.jpg', img)
    counter += 1

# create test dataset
counter = 0
for image in D_test:
    img = cv2.imread(image[1], 1)
    img = cv2.resize(img, (128, 128))
    nume = image[1][13:]
    if nume[0].islower() == True:
        found = nume.partition("_")[2].partition("-")[0]
        born_year = found.partition("_")[2].partition(" ")[0]
        found1 = nume[::-1].partition("gpj.")[2].partition("_")[0]
        photo_year = found1[::-1]
        age_val = int(photo_year) - int(born_year)
    else:
        born_year = nume.partition("_")[2].partition("-")[0]
        found1 = nume[::-1].partition("gpj.")[2].partition("_")[0]
        photo_year = found1[::-1]
        age_val = int(photo_year) - int(born_year)
    cv2.imwrite('dataset1/age/test/' + str(age_val) + '/' + str(counter) + '.jpg', img)
    counter += 1


