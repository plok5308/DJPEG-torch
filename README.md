# DJPEG-torch
This repository contains the dataset and codes for training and validating the Double JPEG Detection model. If you have any questions, feel free to send to my e-mail (plok5308@gmail.com).

If you find this repository helpful, please consider giving it a star ⭐! It helps make this project more visible to others.

- Title: Double JPEG Detection in Mixed JPEG Quality Factors using Deep Convolutional Neural Network
- Conference: The European Conference on Computer Vision 2018 (ECCV2018)

## Requirements
torch, PIL, matplotlib.
```bash
$ pip install torch
$ pip install pillow
$ pip install matplotlib
```

## Dataset downloading
You can download the dataset from Hugging Face:

**Option 1: Direct download from browser**
1. Visit: https://huggingface.co/plok5308/djpeg_dataset
2. Click on "Files and versions"
3. Download the required files

**Option 2: Using git lfs**
```bash
git lfs clone https://huggingface.co/datasets/plok5308/djpeg_dataset
```

## Model weights downloading
The trained model weights are available on Hugging Face:

**Option 1: Direct download from browser**
1. Visit: https://huggingface.co/plok5308/djpegnet
2. Click on "Files and versions"
3. Download the required files

**Option 2: Using git lfs**
```bash
git clone https://huggingface.co/plok5308/djpegnet
```

## Training
Training process from scratch weights. It needs the dataset.
```bash
$ python train.py
```

## Valid
Valid script. It needs the dataset and the network weights. ACC, TPR, TNR are 93.28%, 90.59%, 95.97% respectively.
```bash
$ python valid.py
```

## Application
Detecting splicing and copy-move forgery example. It needs the network weights.
```bash
$ python application.py
```

## Citation
If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{park2018double,
    title     = {Double JPEG detection in mixed JPEG quality factors using deep convolutional neural network},
    author    = {Park, Jinseok and Cho, Donghyeon and Ahn, Wonhyuk and Lee, Heung-Kyu},
    booktitle = {Proceedings of the European conference on computer vision (ECCV)},
    pages     = {636--652},
    year      = {2018}
}
```
