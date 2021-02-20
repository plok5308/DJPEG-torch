# DJPEG-torch

Title: Double JPEG Detection in Mixed JPEG Quality Factors using Deep Convolutional Neural Network
Conference: The European Conference on Computer Vision 2018 (ECCV2018)
Description: This repository has that dataset downloading script, training script, valid script, and application script.

## Dataset downloading
```
$ chmod +x download.sh
$ ./download.sh
$ tar -zxvf jpeg_data.tar.gz
```

## Training
```
$ python train.py [--batch_size int] [--data_path str] [--epoch int] [--loss_interval int] [--split_training]

```

## Valid
$ python valid.py --net_name str

