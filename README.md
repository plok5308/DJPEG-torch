# DJPEG-torch

- Title: Double JPEG Detection in Mixed JPEG Quality Factors using Deep Convolutional Neural Network

- Conference: The European Conference on Computer Vision 2018 (ECCV2018)

- Description: This repository has that dataset downloading script, training script, valid script, and application script.

## Requirements
torch
```
$ pip install torch
```


## Dataset downloading
```
$ chmod +x download.sh
$ ./download.sh
$ tar -zxvf jpeg_data.tar.gz
```
if download.sh cannot download dataset, then please use direct link.

https://drive.google.com/file/d/13sWjSLVTpLftRO3f0wi_ZhupgtgIDBUC/view?usp=sharing



## model weights downloading
```
$ cd model
$ ./download_model.sh
```
if download_mode.sh cannot download the model weights, then please use direct link.

https://drive.google.com/file/d/1OYAqDGovzPn8qTL1zEesbheByt5DUi7o/view?usp=sharing

## Training
```
$ python train.py
```

## Valid
```
$ python valid.py
```
