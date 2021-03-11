import pandas as pd
import numpy as np
from PIL import Image
import cv2
import gc
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
from os import getcwd
import glob

header_list = ['subDirectory_filePath', 'emotion', 'pixels']
pixels = pd.read_csv("affectnet/pixels.csv", names=header_list)
pixels['Usage'] = 'Training'

pixels.shape
pixels.head(2)

pixels = pixels[['emotion', 'pixels', 'Usage']]
pixels.shape
pixels.head(9)

''' convert the categorical labels so that it is compatable with FER2013 '''
pixels.emotion.dtype
dic = {0: "6", 1: "3", 2: "4", 3: "5", 4: "2", 5: "1", 6: "0"}
pixels.replace({"emotion": dic}, inplace=True)
pixels.emotion.dtype
pixels['emotion'] = pixels['emotion'].astype('int64')
pixels.head(9)
pixels.dtypes

''' save the csv file '''
pixels.to_csv('fer2021.csv', index = False, header=True)
#fer_data2021 = pd.read_csv('fer2021.csv')
#fer_data2021.shape
#fer_data2021.tail(3)
