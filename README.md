# Facial Expression Recognition

---

## Overview

This project aims to develop a CNN-based model to recognize Facial Expression of human faces on any custom image or video

## Extract the AffectNet database

- AffectNet contains about `1M (122GB)` facial images collected from the Internet. About ~ `420K (55GB)` images are manually annotated
- Image Properties - The average image resolution of faces in AffectNet are 425 x 425 with STD of **349 x 349** of pixels
- Emotion categories - Eleven annotated emotions are provided: **0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger**, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face, of which we will use the first seven categories
- Run `unrar x -e Manually_Annotated.part01.rar` and all the images will be extracted and combined in the folder **Manually_Annotated_Images**

    - Note that `Manually_Annotated.part01.rar` is the first part of the whole thing and the other parts need to be in the same folder
- There are two labeling csv files for **Manually_Annotated_Images** folder - **training.csv** and **validation.csv**
    - Labeling csv files contain: `subDirectory_filePath`, `face_x`, `face_y`, `face_width`, `face_height`, `facial_landmarks` (68), `expression`, `valence`, `arousal`

## How to use the scripts

- Modify the path and filenames in `process_csv.py` **(base, done, csv_file, new_val_txt)**

    - Change the path  `affectnet/test` to `affectnet/Manually_Annotated_Images`
    - Empty the folder `affectnet/Manually_train_croped`
    - Change the filename `affectnet/test.csv` to `affectnet/training.csv` (`validation.csv`)
    - Change the filename `affectnet/test.txt` to `affectnet/training.txt` (`validation.txt`)
    - Comment the following codes (Perform `cat header.csv validation-copy.csv > validation.csv` for **validation.csv**)
    ```python
    with open(csv_file,newline='') as csvfile:
        r = csv.reader(csvfile)
        data = [line for line in r]
    with open(csv_file,'w',newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal'])
        w.writerows(data)
    ```
    - Comment lines 71&72 [link](https://github.com/chenghanc/Emotion/blob/main/process_csv.py#L71-L72) if we only need path of filename and expression (recommended when collecting cropped images only)
    - Resize the images to **48 x 48** [link](https://github.com/chenghanc/Emotion/blob/main/process_csv.py#L80) when collecting cropped images

- Run [process_csv.py](https://github.com/chenghanc/Emotion/blob/main/process_csv.py) **and collect cropped images**
    - Rename the folder `done` to `base`
    - Copy the file for backup `affectnet/training.txt` (`validation.txt`)
- Run [process_csv_pixels.py](https://github.com/chenghanc/Emotion/blob/main/process_csv_pixels.py) **and generate pixel values of images** in `affectnet/training.txt` (`validation.txt`)
    - Same as above **`affectnet/training.csv` (`validation.csv`)**
    - Same as above **`affectnet/training.txt` (`validation.txt`)**
    - Rename the file `affectnet/training.txt` (`validation.txt`) to `affectnet/pixels.csv`

- Run [process_csv_save.py](https://github.com/chenghanc/Emotion/blob/main/process_csv_save.py) and save training csv file
    - We will get two `fer2021.csv` files for training/validation, which we rename `fer2021-affectnet-train.csv` and `fer2021-affectnet-valid.csv`, respectively
    - Perform `cat fer2021-affectnet-train.csv fer2021-affectnet-valid.csv fer2013.csv > fer2021.csv`

## Visualize images

- We can visualize the images by running [preprocess_csv_affectnet.py](https://github.com/chenghanc/Emotion/blob/main/preprocess_csv_affectnet.py), provided that input files are properly selected

## Train and Evaluate model

- We can train and evaluate model by using this [project](https://github.com/chenghanc/Facial-Expression-Recognition.Pytorch)
- Pre-processing **AffectNet** + **Fer2013** dataset
    - Put **fer2021.csv** in the data folder and change the filename file = 'data/fer2021.csv' in preprocess_fer2013.py, then
    - python preprocess_fer2013.py
    - If everything works fine, it will show 
    ```
    (316110, 2304)
    (3589, 2304)
    (3589, 2304)
    Save data finish!!!
    ```
- Train and Evaluate model
    - Modify line 28 in fer.py **(28709 to 316110)**, then
    - python mainpro_FER.py --model VGG19 --bs 64 --lr 0.01

---

## References

---

<details><summary><b>CLICK ME</b> - References</summary>

- [AffectNet database](http://mohammadmahoor.com/affectnet/)
- [process-AffectNet-csv](https://github.com/yangyuke001/process-AffectNet-csv)
- [FER2013](https://github.com/elzawie/FER2013)
- [Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)
- [Facial Expressions](https://github.com/nikhil-salodkar/facial_expression)
- [DTAN](https://github.com/HayeonLee/DTAN-ICCV15-pytorch)

</details>
