# Facial Expression Recognition

Perform Emotion Classification and Detection using Torch and Darknet

---

## Overview

### Functions

This project aims to develop models to recognize/detect Face Expression of human faces on any custom image or video

- Emotion Classification (Darknet and Torch)
- Emotion Detection (Darknet)

### Requirements

- Python 3.8.5
- Pytorch 1.6.0 (For training classifier)
- h5py (Preprocessing)
- sklearn (Plot confusion matrix)
- OpenCV
- Darknet (For training classifier and detector)

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
    - Comment lines 71&72 [link](https://github.com/chenghanc/Emotion2/blob/main/process_csv.py#L71-L72) if we only need path of filename and expression (recommended when collecting cropped images only)
    - Resize the images to **48 x 48** [link](https://github.com/chenghanc/Emotion2/blob/main/process_csv.py#L80) when collecting cropped images

- Run [process_csv.py](https://github.com/chenghanc/Emotion2/blob/main/process_csv.py) **and collect cropped images**
    - Rename the folder `done` to `base`
    - Copy the file for backup `affectnet/training.txt` (`validation.txt`)
- Run [process_csv_pixels.py](https://github.com/chenghanc/Emotion2/blob/main/process_csv_pixels.py) **and generate pixel values of images** in `affectnet/training.txt` (`validation.txt`)
    - Same as above **`affectnet/training.csv` (`validation.csv`)**
    - Same as above **`affectnet/training.txt` (`validation.txt`)**
    - Rename the file `affectnet/training.txt` (`validation.txt`) to `affectnet/pixels.csv`

- Run [process_csv_save.py](https://github.com/chenghanc/Emotion2/blob/main/process_csv_save.py) and save training csv file
    - We will get two `fer2021.csv` files for training/validation, which we shall rename to `fer2021-affectnet-train.csv` and `fer2021-affectnet-valid.csv`, respectively
    - Perform `cat fer2021-affectnet-train.csv fer2021-affectnet-valid.csv fer2013.csv > fer2021.csv` and we will get **AffectNet** + **Fer2013** training dataset **fer2021.csv**
    - Perform `cat fer2021-affectnet-train.csv fer2021-affectnet-valid-PublicTest.csv fer2021-affectnet-valid-PrivateTest.csv > affectnet.csv` and we will get **AffectNet** training dataset **affectnet.csv**

## Visualize images

- We can visualize the images by running [preprocess_csv_affectnet.py](https://github.com/chenghanc/Emotion2/blob/main/preprocess_csv_affectnet.py), provided that input files are properly selected

## Train and Evaluate model (Torch)

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
    - The training dataset consists of 316,110 examples. The public test dataset consists of 3,589 examples. The private test dataset consists of another 3,589 examples
    - Similarly for **AffectNet** training dataset **affectnet.csv**. The training dataset consists of 283,901 examples. The public test dataset (= private test dataset) consists of 3,500 examples
- Train and Evaluate model
    - Modify line 28 in fer.py **(28709 to 316110)**, then
    - python mainpro_FER.py --model VGG19 --bs 64 --lr 0.01

<details><summary><b>CLICK ME</b> - Settings of parameters used for training</summary>

| Dataset          | Parameters                   | Values                            |
|------------------|:----------------------------:|:---------------------------------:|
| AffectNet        | Size of images used          | 48 x 48                           |
|                  | Optimizer                    | Stochastic Gradient Descent (SGD) |
|                  | Number of epochs             | 200 - 250                         |
|                  | Batch size                   | 64                                |
|                  | Learning rate                | 0.01                              |
|                  | Momentum                     | 0.9                               |
|                  | Learning decay               | 5e-4 (4e-5)                       |
|                  | Start of learning rate decay | After 60  epochs                  |
|                  | Continues decaying           | Every 5   epochs                  |

</details>

- Plot confusion matrix
    - python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest
    - python plot_fer2013_confusion_matrix.py --model VGG19 --split PublicTest
- Visualize for a test image
    - python visualize.py
- **Note:** Before training the network, the faces can also be cropped and resized to **96 x 96** pixels. Just repeat the steps in **How to use the scripts** and modify the resized images to **96 x 96** [link](https://github.com/chenghanc/Emotion2/blob/main/process_csv.py#L80)

## Train and Evaluate model (Darknet) - Data Preparation

<details><summary><b>CLICK ME</b> - Converting AffectNet dataset into YOLO format (For training classifier) </summary>

- Modify the input filename and image folder in process_classifier.py
    - Change the filename affectnet/test.csv to affectnet/training.csv (or validation.csv)
    - Point the path **full_path_to_dataset = '...'** to **Manually_Annotated_Images**
    - Run process_classifier.py

- Make A Dataset Config File (`emotion-classifier.data`)
    - classes=7   : the dataset has 7 different classes
    - train  = ...: where to find the list of training files
    - valid  = ...: where to find the list of validation files
    - labels = ...: where to find the list of possible classes
    - backup = ...: where to save backup weight files during training
    - top    = 5  : calculate top-n accuracy at test time (in addition to top-1)
```ini
classes=7
train  = train.list
valid  = test.list
labels = labels.txt
backup = backup
top=5
```

- Prepare `train.list`, `test.list` and `labels.txt` the same way as detector (described in next section)

- The training dataset consists of **283,901** examples. The validation dataset consists of **3,500** examples

- References: Please visit following links for more information
[Train Classifier on ImageNet (ILSVRC2012)](https://github.com/AlexeyAB/darknet/wiki/Train-Classifier-on-ImageNet-(ILSVRC2012))
[Train a Classifier on CIFAR-10](https://pjreddie.com/darknet/train-cifar/)
[ImageNet Classification](https://pjreddie.com/darknet/imagenet/)

</details>

<details><summary><b>CLICK ME</b> - Converting AffectNet dataset into YOLO format (For training detector)</summary>

- Modify the input filename and image folder in process_od.py
    - Change the filename affectnet/test.csv to affectnet/training.csv (or validation.csv) [link](https://github.com/chenghanc/Emotion2/blob/main/process_od.py#L14)
    - Point the path **full_path_to_dataset = '...'** to **Manually_Annotated_Images** [link](https://github.com/chenghanc/Emotion2/blob/main/process_od.py#L55)
    - We can increase the size of BBox by enlarging width and height 
    - Run process_od.py
    - Add following code if necessary [link](https://github.com/chenghanc/Emotion2/blob/main/process_od.py#L81)
    ```python
    try:
        h, w = image.shape[:2]
    except AttributeError:
        print('NoneType object has no attribute shape')
        pass
    ```

- We can also check if any problematic data exists. Filter output based on file size
    - `cd Manually_Annotated_Images`
    - `find . -maxdepth 2 -size 0 \( -name \*.jpg -o -name \*.png -o -name \*.JPG \) | awk '{print "mv "$1" ../problematic"}' > problematic.sh`
    - `mv ./103/29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg ../problematic`

- We can filter out larger images based on file size
    - `find . -maxdepth 1 -size +385k \( -name \*.jpg \) | awk '{print "mv "$1" ../problematic"}' > problematic.sh`

- Find number of files in a directory
    - `find . -maxdepth 1 -type f -name "*.jpg" | wc -l`
    - `find . -maxdepth 1 -type f -name "*.txt" | wc -l`

- Move large number of files
    - `find Manually_Annotated_Images -maxdepth 1 -name '*.jpg' -exec mv {} anno-train \;`
    - `find Manually_Annotated_Images -name '*.txt' -exec mv {} anno-train \;`

- Create `train.txt`, `test.txt` and `emotion.names`
    - `ls -d "$PWD"/anno-train/*.jpg  > train.txt` (`find "$PWD"/anno-train -name \*.jpg > train.txt`)
    - `ls -d "$PWD"/anno-valid/*.jpg  > test.txt`  (`find "$PWD"/anno-valid -name \*.jpg > test.txt`)
    - Prepare `emotion.names`
    ```ini
    Neutral
    Happy
    Sad
    Surprise
    Fear
    Disgust
    Anger
    ```

- Prepare `emotion.data`
  ```ini
  classes= 7
  train  = train.txt
  valid  = test.txt
  names  = emotion.names
  backup = backup
  ```

- The training dataset consists of **283,901** examples. The validation dataset consists of **3,500** examples

</details>

<details><summary><b>CLICK ME</b> - Settings of parameters used for training detector</summary>

- Batch size: 64 (batch=64)
- Total training data: 283,901
- Iterations: 400,000 (max_batches = 400,000)
- 1 epoch = 283901 / 64 = 4436 iterations
- 400000 x 64 / 283901 = 90 epochs

| Dataset          | Parameters                        | Values                                        |
|------------------|:---------------------------------:|:---------------------------------------------:|
| AffectNet        | Network resolutions               | 256 x 256 (608 x 608)                         |
|                  | Data augmentation                 | saturation, exposure, hue, mosaic, jitter etc |
|                  | Batch size                        | batch=64                                      |
|                  | Optimizator                       | momentum=0.9                                  |
|                  | Optimizator                       | decay=0.0005                                  |
|                  | Optimizator                       | learning_rate=0.00261                         |
|                  | Optimizator                       | burn_in=1000                                  |
|                  | Optimizator                       | max_batches = 400,000                         |
|                  | Number of epochs	                 | 90 - 250                                      |
|                  | Policy for changing learning rate | policy=steps                                  |

</details>

## Train and Evaluate model (Darknet)

- We can train and evaluate **classifier** by using this project [darknet](https://github.com/AlexeyAB/darknet)
    - Training
    ```
    ./darknet classifier train emotion-classifier.data csdarknet53-omega.cfg -topk
    ```

    - Classifictation - Predict
    ```
    ./darknet classifier predict emotion-classifier.data csdarknet53-omega.cfg csdarknet53-omega_last.weights test.jpg
    ```

    - Classifictation - Check accuracy Top1 / Top5
    ```
    ./darknet classifier valid emotion-classifier.data csdarknet53-omega.cfg csdarknet53-omega_last.weights
    ```

- We can train and evaluate **detector** by using this project [darknet](https://github.com/AlexeyAB/darknet)
    - Big model
    ```
    $ ./darknet detector train emotion.data emotion.cfg yolov4.conv.137 -map -dont_show -mjpeg_port 8090 |tee -a trainRecord.txt
    ```

    - Big model - Alternatively, we can use pre-trained weights from Classifictation as a starting point (Use the first 105 layers from the pre-trained weights on AffectNet), which can be obtained by running
    ```
    $ ./darknet partial csdarknet53-omega.cfg csdarknet53-omega_final.weights csdarknet53-omega.conv.105 105
    ```
    - Then
    ```
    $ ./darknet detector train emotion.data emotion.cfg csdarknet53-omega.conv.105 -map -dont_show -mjpeg_port 8090 |tee -a trainRecord.txt
    ```

    - Tiny model
    ```
    $ ./darknet detector train emotion.data emotion-tiny.cfg yolov4-tiny.conv.29 -map -dont_show -mjpeg_port 8090 |tee -a trainRecord.txt
    ```

## Train and Evaluate model (Darknet) Tiny V3

- We can first train a **classifier**
```
$ ./darknet classifier train emotion-classifier.data darknet_ud.cfg -topk -dont_show -mjpeg_port 8090 |tee -a log.txt
```

- The first 11 layers from the pre-trained weights on AffectNet
```
$ ./darknet partial darknet_ud.cfg darknet_ud_final.weights darknet.conv.11 11
```

- Then we can train a **detector**
```
$ ./darknet detector train emotion.data v3-tiny.cfg darknet.conv.11     -map -dont_show -mjpeg_port 8090 |tee -a log.txt

or

$ ./darknet detector train emotion.data v3-tiny.cfg yolov3-tiny.conv.11 -map -dont_show -mjpeg_port 8090 |tee -a log.txt
```

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
