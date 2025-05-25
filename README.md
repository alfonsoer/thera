# Exercice chez Therapanacea
Exercice de classification d’images
The training dataset is composed of 100k images in the directory train_img of the zip file: ml_exercise_therapanacea.zip
In the zip file, there is a label file, label_train.txt. It contains a binary class label for each input image. The name of the images are numbers. These numbers correspond to a line in the label file.
You need to send the following:
  1. The class label for the test images in the directory val_img.
There are 20k images in the test dataset, hence you need to provide a file with 20k
lines, in the same order as in the test dataset. Please name this file “label_val.txt”
  2. Your commented code (either as one or more python files, or a github link or jupyter
notebook).
Note that there is no need to send back the images.
You will be noted not only on the test labels but also on the quality of your code.

# Stage 1 
I've quckly checked the images and the labels. I've tried to elucidate what features are taken into account from the images to tye to chose the best suitable classifier. So first impression: it isn't a genre classification problem. 
# Stage 2 
I'm going to count the samples each class has. Check for imbalanced classes ... 
```text
> python main.py --mode=explore --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --img_test_dir='/home/sagemaker-user/val_img'
There are  100000  images for training
        filename
cohort          
0          12102
1          87898
Percentages: class 0:  12.102  % class 1:  87.898  %
Data is strongly imbalanced
Exploring testing dataset
There are  20000  images for testing
```
# Stage 3
After performing data exploration, I discovered that the classification task involves distinguishing between people based on attributes such as hats, glasses, caps, etc. There are several pre-trained models available for attribute classification [(e.g., FaceAttr-Analysis)](https://github.com/Hawaii0821/FaceAttr-Analysis/tree/master) on the CelebA dataset. However, I decided to take my own approach and train a model from scratch using a very simple architecture. Indeed, another factor to consider when choosing a classifier is the specific application. But in this case, the exercise does not provide further details like the maximum HTER expected on the training set.
# My approach
### 1. Generate predictions on the validation images based on the majority vote of N classifiers.
### 2. Train N classifiers using N-fold cross-validation or subsets. This involves splitting the dataset into N stratified folds to ensure that samples from the minority class are properly represented.
### 3. Each classifier is based on the same CNN architecture, trained on a different fold
### 4. Choose a very simple architecture that includes batch normalization and dropout layers to help reduce overfitting. The performance will be used as a baseline. 
### 5. For data augmentation, I included only slight random shifts and rotations (no more than 5 degrees), since the images consistently depict a portrait of the person.
### 6. Repeat the experiment using ResNet with pretrained weights on ImageNet and remove the last layer to adapt the output. The input should be also adapted to handle 64x64 images. 
### 7. Choose the architechture that gives better results.

# Results
### Results with a simple architecture
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─ModuleList: 1                          []                        --
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-1                  [-1, 16, 64, 64]          448
|    |    └─BatchNorm2d: 3-2             [-1, 16, 64, 64]          32
|    |    └─ReLU: 3-3                    [-1, 16, 64, 64]          --
├─MaxPool2d: 1-1                         [-1, 16, 32, 32]          --
├─ModuleList: 1                          []                        --
|    └─ModuleList: 2                     []                        --
|    |    └─Dropout2d: 3-4               [-1, 16, 32, 32]          --
|    |    └─Conv2d: 3-5                  [-1, 32, 32, 32]          4,640
|    |    └─BatchNorm2d: 3-6             [-1, 32, 32, 32]          64
|    |    └─ReLU: 3-7                    [-1, 32, 32, 32]          --
├─MaxPool2d: 1-2                         [-1, 32, 16, 16]          --
├─ModuleList: 1                          []                        --
|    └─ModuleList: 2                     []                        --
|    |    └─Dropout2d: 3-8               [-1, 32, 16, 16]          --
|    |    └─Conv2d: 3-9                  [-1, 64, 16, 16]          18,496
|    |    └─BatchNorm2d: 3-10            [-1, 64, 16, 16]          128
|    |    └─ReLU: 3-11                   [-1, 64, 16, 16]          --
├─ModuleList: 1                          []                        --
|    └─Dropout: 2-1                      [-1, 16384]               --
|    └─Linear: 2-2                       [-1, 64]                  1,048,640
|    └─ReLU: 2-3                         [-1, 64]                  --
|    └─Dropout: 2-4                      [-1, 64]                  --
|    └─Linear: 2-5                       [-1, 128]                 8,320
|    └─ReLU: 2-6                         [-1, 128]                 --
|    └─Linear: 2-7                       [-1, 1]                   129
==========================================================================================
Total params: 1,080,897
Trainable params: 1,080,897
Non-trainable params: 0
Total mult-adds (M): 12.26
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 1.75
Params size (MB): 4.12
Estimated Total Size (MB): 5.92
==========================================================================================
```

### Results with ResNet with pretrained weights 
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 64, 127, 127]        9,408
├─BatchNorm2d: 1-2                       [-1, 64, 127, 127]        128
├─ReLU: 1-3                              [-1, 64, 127, 127]        --
├─MaxPool2d: 1-4                         [-1, 64, 64, 64]          --
├─Sequential: 1-5                        [-1, 64, 64, 64]          --
|    └─BasicBlock: 2-1                   [-1, 64, 64, 64]          --
|    |    └─Conv2d: 3-1                  [-1, 64, 64, 64]          36,864
|    |    └─BatchNorm2d: 3-2             [-1, 64, 64, 64]          128
|    |    └─ReLU: 3-3                    [-1, 64, 64, 64]          --
|    |    └─Conv2d: 3-4                  [-1, 64, 64, 64]          36,864
|    |    └─BatchNorm2d: 3-5             [-1, 64, 64, 64]          128
|    |    └─ReLU: 3-6                    [-1, 64, 64, 64]          --
|    └─BasicBlock: 2-2                   [-1, 64, 64, 64]          --
|    |    └─Conv2d: 3-7                  [-1, 64, 64, 64]          36,864
|    |    └─BatchNorm2d: 3-8             [-1, 64, 64, 64]          128
|    |    └─ReLU: 3-9                    [-1, 64, 64, 64]          --
|    |    └─Conv2d: 3-10                 [-1, 64, 64, 64]          36,864
|    |    └─BatchNorm2d: 3-11            [-1, 64, 64, 64]          128
|    |    └─ReLU: 3-12                   [-1, 64, 64, 64]          --
├─Sequential: 1-6                        [-1, 128, 32, 32]         --
|    └─BasicBlock: 2-3                   [-1, 128, 32, 32]         --
|    |    └─Conv2d: 3-13                 [-1, 128, 32, 32]         73,728
|    |    └─BatchNorm2d: 3-14            [-1, 128, 32, 32]         256
|    |    └─ReLU: 3-15                   [-1, 128, 32, 32]         --
|    |    └─Conv2d: 3-16                 [-1, 128, 32, 32]         147,456
|    |    └─BatchNorm2d: 3-17            [-1, 128, 32, 32]         256
|    |    └─Sequential: 3-18             [-1, 128, 32, 32]         8,448
|    |    └─ReLU: 3-19                   [-1, 128, 32, 32]         --
|    └─BasicBlock: 2-4                   [-1, 128, 32, 32]         --
|    |    └─Conv2d: 3-20                 [-1, 128, 32, 32]         147,456
|    |    └─BatchNorm2d: 3-21            [-1, 128, 32, 32]         256
|    |    └─ReLU: 3-22                   [-1, 128, 32, 32]         --
|    |    └─Conv2d: 3-23                 [-1, 128, 32, 32]         147,456
|    |    └─BatchNorm2d: 3-24            [-1, 128, 32, 32]         256
|    |    └─ReLU: 3-25                   [-1, 128, 32, 32]         --
├─Sequential: 1-7                        [-1, 256, 16, 16]         --
|    └─BasicBlock: 2-5                   [-1, 256, 16, 16]         --
|    |    └─Conv2d: 3-26                 [-1, 256, 16, 16]         294,912
|    |    └─BatchNorm2d: 3-27            [-1, 256, 16, 16]         512
|    |    └─ReLU: 3-28                   [-1, 256, 16, 16]         --
|    |    └─Conv2d: 3-29                 [-1, 256, 16, 16]         589,824
|    |    └─BatchNorm2d: 3-30            [-1, 256, 16, 16]         512
|    |    └─Sequential: 3-31             [-1, 256, 16, 16]         33,280
|    |    └─ReLU: 3-32                   [-1, 256, 16, 16]         --
|    └─BasicBlock: 2-6                   [-1, 256, 16, 16]         --
|    |    └─Conv2d: 3-33                 [-1, 256, 16, 16]         589,824
|    |    └─BatchNorm2d: 3-34            [-1, 256, 16, 16]         512
|    |    └─ReLU: 3-35                   [-1, 256, 16, 16]         --
|    |    └─Conv2d: 3-36                 [-1, 256, 16, 16]         589,824
|    |    └─BatchNorm2d: 3-37            [-1, 256, 16, 16]         512
|    |    └─ReLU: 3-38                   [-1, 256, 16, 16]         --
├─Sequential: 1-8                        [-1, 512, 8, 8]           --
|    └─BasicBlock: 2-7                   [-1, 512, 8, 8]           --
|    |    └─Conv2d: 3-39                 [-1, 512, 8, 8]           1,179,648
|    |    └─BatchNorm2d: 3-40            [-1, 512, 8, 8]           1,024
|    |    └─ReLU: 3-41                   [-1, 512, 8, 8]           --
|    |    └─Conv2d: 3-42                 [-1, 512, 8, 8]           2,359,296
|    |    └─BatchNorm2d: 3-43            [-1, 512, 8, 8]           1,024
|    |    └─Sequential: 3-44             [-1, 512, 8, 8]           132,096
|    |    └─ReLU: 3-45                   [-1, 512, 8, 8]           --
|    └─BasicBlock: 2-8                   [-1, 512, 8, 8]           --
|    |    └─Conv2d: 3-46                 [-1, 512, 8, 8]           2,359,296
|    |    └─BatchNorm2d: 3-47            [-1, 512, 8, 8]           1,024
|    |    └─ReLU: 3-48                   [-1, 512, 8, 8]           --
|    |    └─Conv2d: 3-49                 [-1, 512, 8, 8]           2,359,296
|    |    └─BatchNorm2d: 3-50            [-1, 512, 8, 8]           1,024
|    |    └─ReLU: 3-51                   [-1, 512, 8, 8]           --
├─AdaptiveAvgPool2d: 1-9                 [-1, 512, 1, 1]           --
├─Sequential: 1-10                       [-1, 1]                   --
|    └─Linear: 2-9                       [-1, 128]                 65,664
|    └─BatchNorm1d: 2-10                 [-1, 128]                 256
|    └─ReLU: 2-11                        [-1, 128]                 --
|    └─Dropout: 2-12                     [-1, 128]                 --
|    └─Linear: 2-13                      [-1, 1]                   129
==========================================================================================
Total params: 11,242,561
Trainable params: 11,242,561
Non-trainable params: 0
Total mult-adds (G): 2.39
==========================================================================================
Input size (MB): 0.74
Forward/backward pass size (MB): 49.25
Params size (MB): 42.89
Estimated Total Size (MB): 92.88
==========================================================================================
```

# How to execute the code ?
## Data exploration
```
python main.py --mode=explore --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --img_test_dir='/home/sagemaker-user/val_img'
```
## CNN training
```
python main.py --mode=train --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --epochs=15 --lr=0.001 --step=5 --gamma=0.7 --save_dir='/home/sagemaker-user/thera/results'
```
## CNN testing
```
python main.py --mode=test --img_test_dir='/home/sagemaker-user/val_img' --model_dir='/home/sagemaker-user/thera/results/models'
```

# Appendix
## Training plots for each fold.

## N-classifiers predictions and majority vote

# Off-topic
Since my personal laptop can't handle ML model training (I used to use a 4-GPU server at the lab), I initially thought about using Google Colab. However, I'm not very enthusiastic of launching code in chunks—at least not unless I’m in debugging mode besides the fact that data transfer took an eternity. So I later switched to Amazon SageMaker and created a JupyterLab environment with a 16 GB GPU and a 4-core CPU. The interface for writing and running scripts is very simple, user-friendly, and runs smoothly. I really enjoyed setting up my AWS SageMaker environment.
