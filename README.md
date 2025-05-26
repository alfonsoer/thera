# Exercice
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

# Deliverables
The code can be found here: [(Theraclassifier)](https://github.com/alfonsoer/thera/tree/main/TheraClassifier) and the output label_val.txt file that is generated in the same directory level as val_img is here [(label_val.txt)](https://github.com/alfonsoer/thera/blob/main/label_val.txt)

# Procedure
## Data exploration
### Stage 1
I've quckly checked the images and the labels. I've tried to elucidate what features are taken into account from the images to tye to chose the best suitable classifier. So first impression: it isn't a genre classification problem. 
### Stage 2 
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
### Stage 3

After performing data exploration, I discovered that the classification task involves distinguishing between people based on attributes such as hats, glasses, caps, etc. There are several pre-trained models available for attribute classification [(e.g., FaceAttr-Analysis)](https://github.com/Hawaii0821/FaceAttr-Analysis/tree/master) on the CelebA dataset. However, I decided to take my own approach and train a model from scratch using a very simple architecture. Indeed, another factor to consider when choosing a classifier is the specific application. But in this case, the exercise does not provide further details like a typical HTER expected on the training set.
## Method
1. Generate predictions on the validation images based on the majority vote of N classifiers.
2. Train N classifiers using N-fold cross-validation. This involves splitting the dataset into N stratified folds to ensure that samples from the minority class are properly represented in each subset.
3. Each classifier is based on the same CNN architecture, trained on a different fold.
4. Choose a very simple architecture that includes batch normalization and also add dropout layers to help reduce overfitting. The performance will be used as a baseline.
5. For data augmentation, I've included only slight random shifts and rotations (no more than 5 degrees), since the images consistently depict a portrait of the person.
6. Repeat the experiment using ResNet with pretrained weights on ImageNet and remove the last layer to adapt the output. The input should be also adapted to handle 64x64 images.
7. Choose the architechture that gives better results.
8. Repeat the experiment adding a Weighted Random Sampler, to also ensure that samples from the minority class are properly represented in each batch

## Results
### Results with a simple architecture
In summary the architecture is composed of sucesive convolutional + batchnorm + relu + dropout layers. The number and size of the kernels are given bellow:
```
conv_layer_widths = [[(16,3)],[(32,3),(32,3)], [(64,3),(64,3)], [(128,3),(128,3)]]
linear_layer_widths = [64,128, 1]
```
The training parameters were:
```
    epochs = 50
    lr=1e-4
    step_size=20
    gamma=0.7
    batch_size = 32
    dropout 2D and 1D = 0.25
    wrs = True #Weighted random sampler ON with replacement ON
```
The performance of the method evaluated by crossvalidation on 5 validation folds is shown in Figure 1.
![Results](https://github.com/user-attachments/assets/8f921480-dfaa-4797-9c06-6642545d7b37)
<h4 align="center">Figure 1</h1>

The label_val.txt validation file is generated in the same directory level as val_img. [(label_val.txt)](https://github.com/alfonsoer/thera/blob/main/label_val.txt)

### Results with ResNet with pretrained weights 
Due to hardware constrains I could not finish testing the pre-trained resnet. The only results I got didn't use weighted random sampling, reaching a BA in the order of 85%. I am pretty sure the results could improve by adding the WRS when training each batch. In general the learning behaviour using this pre-trained renest-18 is more stable accross epochs than the simple CNN I've used before. Pending to finish testing resnet-18 with WRS.

![image](https://github.com/user-attachments/assets/57ea0fec-4e06-459e-a9ee-83958fd7c706)

## How to execute the code ?
### Data exploration
To explore the data content use a command as the following. 
```
python main.py --mode=explore --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --img_test_dir='/home/sagemaker-user/val_img'
```
### CNN training
For tranining, there are some custom parameters we can pass, such as epochs, learning rate, etc as well as the destination folder. Models are saved into the destination path under a folder 'models'. The log folder organizes training-validation plots and stores the metrics history.
```
python main.py --mode=train --img_train_dir='/home/sagemaker-user/train_img' --labels_txt='/home/sagemaker-user/label_train.txt' --epochs=50 --lr=1e-4 --step=20 --gamma=0.7 --save_dir='/home/sagemaker-user/thera/results_vbd_wrs_lr1e-4_50_epochs'
```
### CNN testing
Testing requires the testing path of the data and the model's path. Recall all models are saved into a folder named 'models' which is inside the destination's path that was indicated when training. The label_val.txt file is generated in the same directory level as val_img.
```
python main.py --mode=test --img_test_dir='/home/sagemaker-user/val_img' --model_dir='/home/sagemaker-user/thera/results/models'
```

# Appendix
## N-classifiers predictions and majority vote
Interestingly all indivividual classifiers performed similarly :) See the full table here : [(N-predictions table)](https://github.com/alfonsoer/thera/blob/main/test_predictions_majority_voting.csv)
| filename   	| pred_1 	| pred_2 	| pred_3 	| pred_4 	| pred_5 	| prediction 	|
|------------	|--------	|--------	|--------	|--------	|--------	|------------	|
| 000001.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000002.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000003.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000004.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000005.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000006.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000007.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000008.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000009.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000010.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000011.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000012.jpg 	| 1      	| 0      	| 1      	| 1      	| 1      	| 1          	|
| 000013.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000014.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000015.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000016.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000017.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000018.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000019.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000020.jpg 	| 0      	| 0      	| 0      	| 0      	| 0      	| 0          	|
| 000021.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000022.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000023.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000024.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000025.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000026.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000027.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000028.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|
| 000029.jpg 	| 1      	| 1      	| 1      	| 1      	| 1      	| 1          	|

# Off-topic
Since my personal laptop can't handle ML model training (I used to use a 4-GPU server at the lab), I initially thought about using Google Colab. However, I'm not very enthusiastic of launching code in chunks—at least not unless I’m in debugging mode besides the fact that data transfer took an eternity. So I later switched to Amazon SageMaker and created a JupyterLab environment with a 16 GB GPU and a 4-core CPU. The interface for writing and running scripts is very simple, user-friendly, and runs smoothly. I really enjoyed setting up my AWS SageMaker environment.
