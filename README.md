# DJPEG-torch
This repository has that dataset downloading script, training script, valid script, and application script about the below research paper. If you have question, feel free to send to my e-mail (plok5308@gmail.com).

- Title: Double JPEG Detection in Mixed JPEG Quality Factors using Deep Convolutional Neural Network

- Conference: The European Conference on Computer Vision 2018 (ECCV2018)

## Requirements
torch, PIL, matplotlib.
```
$ pip install torch
$ pip install pillow
$ pip install matplotlib
```

## Dataset downloading
```
$ chmod +x download.sh
$ ./download.sh
$ tar -zxvf jpeg_data.tar.gz
```
if download.sh dosen't download dataset, then please use direct link.

https://drive.google.com/file/d/13sWjSLVTpLftRO3f0wi_ZhupgtgIDBUC/view?usp=sharing

## Model weights downloading
```
$ cd model
$ ./download_model.sh
```
if download_mode.sh doesn't download the model weights, then please use direct link.

https://drive.google.com/file/d/1OYAqDGovzPn8qTL1zEesbheByt5DUi7o/view?usp=sharing

## Training
Training process from scratch weights. It needs the dataset.
```
$ python train.py
```

## Valid
Valid script. It needs the dataset and the network weights. ACC, TPR, TNR are 93.28%, 90.59%, 95.97% respectively.
```
$ python valid.py
```

## Application
Detecting splicing and copy-move forgery example. It needs the network weights.
```
$ python application.py
```

