# CMAL-Net

Code release for 'Learn from Each Other to Classify Better: Cross-layer Mutual Attention Learning for Fine-grained Visual Classification'

![enter image description here](https://github.com/Dichao-Liu/CMAL/blob/main/CMAL.png)
 

### Environment

This source code was tested in the following environment:

Python = 3.7.11

PyTorch = 1.8.0

torchvision = 0.9.0

Ubuntu 18.04.5 LTS

NVIDIA GeForce RTX 3080 Ti

### Dataset

* (1) Download the FGVC-Aircraft, Stanford Cars, and Food-11 datasets and organize the structure as follows:
```
dataset folder
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```
* (2) modify the path to the dataset folders.

### Dependencies

* (1) Inplace-ABN

Install `Inplace-ABN` following the instructions:

https://github.com/Alibaba-MIIL/TResNet/blob/master/requirements.txt

https://github.com/Alibaba-MIIL/TResNet/blob/master/INPLACE_ABN_TIPS.md

* (2) TResNet_L

Download the folder `src` from https://github.com/Alibaba-MIIL/TResNet and save it as:
```
Code
├── basic_conv.py
├── utils.py
├── train_Stanford_Cars_TResNet_L.py
├── ...
├── src
```

### Training

Run the scripts for training, such as `python train_Stanford_Cars_TResNet_L.py`.
