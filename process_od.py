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
emotions_names = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger', 7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'}
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
# fer2021_data['face_x'] = fer2021_data['XMin'] and fer2021_data['face_y'] = fer2021_data['YMin']
fer2021_data['center x'] = ''
fer2021_data['center y'] = ''
fer2021_data['center x'] = fer2021_data['face_x'] + fer2021_data['face_width'] / 2
fer2021_data['center y'] = fer2021_data['face_y'] + fer2021_data['face_height']/ 2

fer2021_data.head(3)

r2 = fer2021_data.loc[:, ['subDirectory_filePath',
                        'expression',
                        'center x',
                        'center y',
                        'face_width',
                        'face_height']].copy()
r2.head(3)
r2.rename(columns={'subDirectory_filePath': 'ImageID', 'expression': 'CategoryID', 'face_width': 'width', 'face_height': 'height'}, inplace=True)
r2

r = r2.loc[r2['CategoryID'] < 7]
r

os.getcwd()

full_path_to_dataset = '/Users/chung/NECHK-Results/github/Emotion2/affectnet/test'

os.chdir(full_path_to_dataset)

r.head()

file_types = ('.jpg', '.JPG', '.png')
num = 0
for root, subdirectories, files in os.walk('.'):

    # Printing sub-directory
    for subdirectory in subdirectories:
        print("Sub Directory:", os.path.join(root, subdirectory) + "\n")

    # Going through all files
    for file in files:
        num += 1

        # Checking if filename ends with ('.jpg', '.JPG', '.png', ...)
        if file.endswith(file_types):
            print("Sub Filenames:", os.path.join(root, file))

            # Reading images
            image = cv2.imread(os.path.join(root, file))

            # Getting real width and height
            h, w = image.shape[:2]
            print("Height & Width", (h,w))
            print("Filenames:    ", file)

            image_name = file[:-4]
            print("Image Names:  ", image_name + "\n")

            # Create separate dataFrame
            sub_r = r.loc[r['ImageID'] == file].copy()

            # Normalizing calculated bounding boxes coordinates
            sub_r['center x'] = sub_r['center x'] / w
            sub_r['center y'] = sub_r['center y'] / h
            sub_r['width'] = sub_r['width'] / w
            sub_r['height'] = sub_r['height'] / h

            # Create separate dataFrame
            resulted_frame = sub_r.loc[:, ['CategoryID',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            if resulted_frame.isnull().values.all():
                continue

            path_to_save = full_path_to_dataset + '/' + image_name + '.txt'

            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

            path_to_save = full_path_to_dataset + '/' + image_name + '.jpg'

            cv2.imwrite(path_to_save, image)

print(num)
