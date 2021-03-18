import pandas as pd
import cv2
import os
from os import getcwd

# AffectNet - YOLO Format
'''
fer2021_data = pd.read_csv('affectnet/test.csv',
                  names=['subDirectory_filePath', 'face_x', 'face_y', 'face_width',
                  'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal'],
                  sep=',')
'''
fer2021_data = pd.read_csv('affectnet/test.csv')

fer2021_data.columns
fer2021_data.shape
emotions_names = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt', 8: 'none', 9: 'uncertain', 10: 'no-face'}
fer2021_data['emotion_name'] = fer2021_data['expression'].map(emotions_names)
fer2021_data.shape
fer2021_data.index
fer2021_data.tail(3)
fer2021_data.sample(n=3)
fer2021_data['expression'].unique()
fer2021_data.emotion_name.value_counts()
fer2021_data.dtypes
fer2021_data.isna().any()
#
fer2021_data['subDirectory_filePath'] = fer2021_data['subDirectory_filePath'].str.split('/').str[1]
fer2021_data.head(3)
#
fer2021_data['ImageID0'] = ''
fer2021_data['ImageID1'] = ''
fer2021_data['ImageID2'] = ''
fer2021_data['ImageID_Classifier'] = ''

new = fer2021_data['emotion_name'].copy()

fer2021_data['ImageID0'] = fer2021_data['subDirectory_filePath'].str.split('.').str[0]
fer2021_data['ImageID1'] = fer2021_data['subDirectory_filePath'].str.split('.').str[1]
fer2021_data['ImageID2'] = fer2021_data['ImageID0'].str.cat(new, sep ="_")
fer2021_data['ImageID_Classifier'] = fer2021_data['ImageID2'].str.cat(fer2021_data['ImageID1'], sep ='.')

fer2021_data
print(fer2021_data['ImageID0'][0])
print(fer2021_data['ImageID1'])
print(fer2021_data['ImageID2'][0])
print(fer2021_data['ImageID_Classifier'][0])

r2 = fer2021_data.loc[:, ['ImageID_Classifier', 'subDirectory_filePath',
                        'expression',
                        'emotion_name',
                        'face_width',
                        'face_height']].copy()
r2.head(3)
r2.rename(columns={'subDirectory_filePath': 'ImageID', 'expression': 'CategoryID', 'emotion_name': 'NameID', 'face_width': 'width', 'face_height': 'height'}, inplace=True)
r2

r = r2.loc[r2['CategoryID'] < 7]
r

# Getting the current directory
os.getcwd()

full_path_to_dataset = '/home/nechk/NECHK-Results/helmet2/emotion/Emotion2/affectnet/test'

# Changing the current directory
# to one with images
os.chdir(full_path_to_dataset)

#file_types = ('.jpg', '.JPG', '.png')
num = 0
for root, subdirectories, files in os.walk('.'):

    # Printing sub-directory
    for subdirectory in subdirectories:
        print("Sub Directory:", os.path.join(root, subdirectory) + "\n")

    # Going through all files
    for file in files:
        num += 1

        # Checking if filename ends with ('.jpg', '.JPG', '.png', ...)
        #if file.endswith(file_types):
        print("Sub Filenames:", os.path.join(root, file))

        # Reading images
        image = cv2.imread(os.path.join(root, file))

        # Getting real width and height
        try:
            h, w = image.shape[:2]
        except AttributeError:
            print('NoneType object has no attribute shape')
            pass
        print("Height & Width", (h,w))
        print("Filenames:    ", file)

        # Slicing only name of the file without extension
        image_name = file[:-4]
        print("Image Names:  ", image_name + "\n")

        # Create separate dataFrame
        sub_r = r.loc[r['ImageID'] == file].copy()

        # Create separate dataFrame
        resulted_frame = sub_r.loc[:, ['ImageID_Classifier', 'CategoryID',
                                        'NameID',
                                        'width',
                                        'height']].copy()

        # Checking if there is no any annotations for current image
        if resulted_frame.isnull().values.all():
            # Skipping this image
            continue

        # Preparing path where to save jpg image
        #for i in resulted_frame.CategoryID:
        for i in resulted_frame.NameID:
            print(i)
            path_to_save = full_path_to_dataset + '/' + image_name + '_' + str(i) + '.jpg'

        # Saving image in jpg format by OpenCV function
        cv2.imwrite(path_to_save, image)

print(num)
