# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 04:00:55 2021

@author: vedhus
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np



def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def preprocess(img, labels):
    img = np.array(img)
    lab_1hot = np.zeros((img.shape[0],img.shape[1],len(labels)), dtype=np.bool)
    for i in range(len(labels)):
        lab_1hot[:,:,i] = img==labels[i]
    return lab_1hot


def gt2kaggle(label_path,csv_read, write_path, csv_write,labelObj,formatType):
    # read a csv file associated with the testing data and create Kaggle solution csv file
    csv_write = csv_write.format(labelObj.name)
    label_path = label_path.format(labelObj.name)
    labels = labelObj.labels
    csvName = os.path.join(csv_read)
    df = pd.read_csv(csvName,header=None)
    labFiles = list(df[0])

    data = pd.DataFrame()
    k = 0
    for labelFile in labFiles:        
       
        labelImages = preprocess(Image.open(os.path.join(label_path,labelFile)),labels)
        k+=1
        print('{2} {0}: {1}'.format(k, labelFile, labelObj.name))
        for j in range(labelImages.shape[2]):
             temp = pd.DataFrame.from_records([
                {
                    'ImageId': labelFile+'_'+str(labels[j]),
                    'EncodedPixels': mask2rle(labelImages[:,:,j])
                }])
             data = pd.concat([data, temp],ignore_index=True)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    data.to_csv(os.path.join(write_path,csv_write), index=False)


class labelClass:
    def __init__(self,name, labels):
        self.name = name
        self.labels = labels
        
# Comment out any labelClass objects for which csv file will not be produced
labelArray = []
labelArray.append(labelClass('component',[0,1,2,3,4,5,6]))
# labelArray.append(labelClass('crack',[1]))
# labelArray.append(labelClass('spall',[1]))
# labelArray.append(labelClass('rebar',[1]))
# labelArray.append(labelClass('ds',[0,1,2,3]))


if __name__ == '__main__':    
    csv_read = 'test.csv'
    label_path = "label\{0}"
    write_path = 'kaggle'
    csv_write_sub = 'submission_{0}.csv'
    
    for labelObj in labelArray:
        gt2kaggle(label_path,csv_read,write_path, csv_write_sub,labelObj,'Submission')
        
    
    
    
    
