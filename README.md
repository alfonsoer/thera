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

### Results with ResNet with pretrained weights 

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
