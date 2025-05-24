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
I've quckly checked the images and the labels. I've tried to elucidate what features are taken into account from the images to tye to chose the best classifier. So first impression: it isn't a genre classification problem. 
# Stage 2 
I'm going to count the samples each class has. Check for imbalanced classes ... 
# Stage 3

# How to execute
## Data exploration
python main.py --mode=explore --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --img_test_dir='/home/sagemaker-user/val_img'
## CNN training
python main.py --mode=train --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --epochs=15 --lr=0.001 --step=5 --gamma=0.7 --save_dir='/home/sagemaker-user/thera/results'
