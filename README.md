**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figure_1.png "Validation without dropout" 
[image2]: ./figure_2.png "Validation with dropout"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `network.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results
* `run1.mp4` with movie from autonomous lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `network.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Model is based on NVIDIA architecture with some changes. At the beggining of the model the network is normalizing the picures (line 64) and rejecting the irrelevant part of the pictures (line 65). There are three covolutional layers with (5x5) window (lines 66, 69, 70) and two with (3x3) window (lines 71, 72).
Next, the four fully conected layers are applied. After first and last convolution (lines 67, 75) the Dropout layer is implement to prevent from overfittiong.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (network.py lines 67, 75). 

The train test was splitted to create validation samples (line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Below there are two charts with validation rated before and after apllying the dropout layers.

##### Before applying the dropout layers:
![][image1]
##### After applying the dropout layers:
![][image2]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (network.py line 81).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used an example data set with additional recordings to improve the behavior of the car. 
Additionally recorded data was including recovery from the sidaways and thight turns passing.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA structure. I thought this model might be appropriate because is is being used for real autonomous driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I also used images from side cameras and filipped central camera images to generalize the model.

To combat the overfitting, I modified the model and add the Dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so I additionally recorded new data to improve the driving behavior in these cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (network.py lines 63-79) consisted of a following layers:

* Lambda layer, output size (160, 320, 3)
* Cropping layer, output size (85, 320, 3)
* Convolutional layer, output size (41, 158, 24)
* Dropout layer
* Relu activation
* Convolutional layer with relu activation, output size (19, 77, 36)
* Convolutional layer with relu activation, output size (8, 37, 48)
* Convolutional layer with relu activation, output size (3, 18, 64)
* Convolutional layer, output size (1, 8, 64)
* Flatten layer, output size 512
* Fully connected layer, output size 100
* Dropout layer
* Relu activation
* Fully connected layer, output size 50
* Fully connected layer, output size 10
* Fully connected layer, output size 1

#### 3. Creation of the Training Set & Training Process

Originally I was creating my own training data set, but I switched to the example one and rerecorded some parts of the route.

After first run after final architecture implementation I saw that car poorly behaves on sharp turns. Also recovery from the side of the tracks is not as good as I would want. I appended a new data to the old dataset and retrained the network again.

The validation set helped determine if the model was over or under fitting. SI used an adam optimizer so that manually training the learning rate wasn't necessary.
