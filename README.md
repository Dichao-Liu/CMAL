# CMAL-Net
 
 
### Tested on
 
Python 3.8.13

PyTorch 1.12.0

torchvision 0.13.0

### Usage

1. Organize datatsets as follows:
```
dataset
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

2. Modify the path to the dataset in the training scripts.

3. Run the script, e.g.:
```
python train_cars_resnet50.py
```
