'''
Ref. https://github.com/yangyuke001/process-AffectNet-csv
'''
from __future__ import print_function
import io
import os
import sys
import random
import cv2
import pandas as pd
from PIL import Image
import csv
import os
from os import getcwd

wd = getcwd()
base = wd + '/affectnet/test'
done = wd + '/affectnet/Manually_train_croped/'
csv_file = wd + '/affectnet/test.csv'
new_val_txt = open(wd + '/affectnet/test.txt','w')

'''
# https://stackoverflow.com/questions/28162358/append-a-header-for-csv-file/51726481
with open(csv_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal'])
'''

# Append a Header for CSV file
#with open(csv_file,newline='') as csvfile:
#    r = csv.reader(csvfile)
#    data = [line for line in r]
#with open(csv_file,'w',newline='') as csvfile:
#    w = csv.writer(csvfile)
#    w.writerow(['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal'])
#    w.writerows(data)
''' Or cat affectnet/header.csv affectnet/test.csv > test.csv '''
# Append a Header for CSV file

fname = []
face_x = []
face_y = []
face_width = []
face_height = []
expression = []
num = 0
with open(csv_file,'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        num += 1
        fname = row['subDirectory_filePath']
        x = int(row['face_x'][0:])
        y = int(row['face_y'][0:])
        width = int(row['face_width'][0:])
        height = int(row['face_height'][0:])
        expression = int(row['expression'][0:])
        floder_dir = fname.split('/')[0]
        img = fname.split('/')[1]
        #crop images
        image = cv2.imread(os.path.join(base,fname))

        #convert image to a string of pixel values
        #image = Image.open(os.path.join(base,fname))
        #img_str = str(list(image.getdata()))
        #img_str2= img_str.strip('[]').replace(',','')
        #write name & expression & pixel values to new txt
        if expression < 7:
            new_val_txt.write(fname)
            new_val_txt.write(',')
            new_val_txt.write(str(expression))
            #new_val_txt.write(',')
            #new_val_txt.write(str(img_str2))
            new_val_txt.write('\n')

        #process img
        try:
            imgROI = image[x:x + width, y:y + height]
        except:
            pass
        imgROI = cv2.resize(imgROI, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
        if not os.path.isdir('affectnet/Manually_train_croped/' + floder_dir):
            os.mkdir('affectnet/Manually_train_croped/' + floder_dir)
        cv2.imwrite(done + floder_dir + '/' + img, gray)
        print(fname)
        cv2.waitKey(0)

    print(num)

'''
# For checking purpose
# Correct
#img = Image.open('affectnet/Manually_Annotated_Images/1176/94f5564aed80a7806ffda40e781e4b5a5e72cfcad47c86e48c1c4f99.jpg','r')
img = Image.open('affectnet/Manually_Annotated_Images/1176/94f5564aed80a7806ffda40e781e4b5a5e72cfcad47c86e48c1c4f99.jpg')
pix_val = list(img.getdata())
pix_val
pixels_values = pd.DataFrame(pix_val, dtype=int)
pixels_values

# Wrong Way
img = cv2.imread('affectnet/Manually_Annotated_Images/1176/94f5564aed80a7806ffda40e781e4b5a5e72cfcad47c86e48c1c4f99.jpg')
#convert image to a string of pixel values
str_rep = str(img.flatten().tolist())
img_str = str_rep.strip('[]').replace(',','')
pixels_values = img_str.split(" ")
pixels_values = pd.DataFrame(pixels_values, dtype=int)
pixels_values
'''
