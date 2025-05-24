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
After performing data exploration I've discovered the classifier task is to classify between persons having or not attributes like hats, glasses, cap, etc
There exist several pre-trained models for attributes classification like this, https://github.com/Hawaii0821/FaceAttr-Analysis/tree/master, however I've decided to go in my own and to train a model from zero with a very simple architechture. Indeed, another factor for choosing a classifier is the application. But in this case, the exercice does not give more details. 
# My approach
## 1. Generate a prediction on the validation images based on the mojority vote of N classifiers. 
## 2. Train N classifiers based on N-folds or subsets. Thus, splitting the DS into N stratified folds to assure samples of the minority class are considered.
## 3. Each classifier will be based on the same CNN trained on each different fold. In this case I have chosen a very simple architechture including batch normalization and dropouts to help reduce overfitting. For data augmentation I have chosen to only inlcude some random shifts and rotation, no more than 10 degrees since images have always a portrait of the person.

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

#Appendix
## Training plots for each fold.

## N-classifiers predictions and majority vote
