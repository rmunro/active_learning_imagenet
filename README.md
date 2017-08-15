# Active Learning with TensorFlow and ImageNet
 Coding exercise to extend tensorflow for imagenet to active learning, for use in a job interview or similar. 
 
The code runs, but IS DELIBERATELY A BAD IMPLEMENTATION OF ACTIVE LEARNING 


# Context 

A company has approached you, that wants to classify a large set of sports images according to the ImageNet set of labels. 

However, there are several problems:
 1. The classification via TensorFlow is not very accurate, so they want to label some of their images to retrain the classifier with new training items.
 1. They have millions of images, so they cannot label them all.
 1. Their collection of images is may include some that are not related to sports, which they don't care about.

This is a real-world situation that occurs regularly. For this exercise, we will use the open set of sports images from CrowdFlower's Data for Everyone program:
https://www.crowdflower.com/data-for-everyone/ 
 
## Getting started 

A (250MB) subset of the images is available at:
http://www.robertmunro.com/research/test_images.tar 

The company made starter code that is in this same directory at:
`active_learning_for_images.py`

# Exercise 
 
Your exercise is to improve the code so that it: 
 1. takes a collection of unlabled images 
 1. attempts to classify each image with Inception trained on ImageNet (2012) 
 1. orders the images according to how each should be manually labled, to create the best possible training data from the new images. 
 
Steps 1 and 2 are implemented (but could be improved). 
 
Step 3 currently orders the images from the most-to-least confidently classified, which is a bad strategy. 

Most of the code to be edited is within `order_images_for_active_learning()`, but you may edit any code that you think will improve the output.

# Potential Solutions
 
There are many extensions to this code, from a 1 hour exercise to improve how confidence is used, to a multiple week exercice that included retraining all/\
parts of the model and provided interfaces that were optimal for retraining. 

## 2 hour Coding Exercise




# Installation and code source

A (250MB) subset of the images is available at:
http://www.robertmunro.com/research/test_images.tar 

The company made starter code that is in this same directory at:
`active_learning_for_images.py`
 
The code is in the style of TensorFlow tutorials and is adapted with thanks to the original authors of: 
https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py 
 
The code can be run in the same tutorial folder (although not required): 
 https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet 
 
To install TensorFlow and for more context on this problem, see: 
 https://www.tensorflow.org/tutorials/image_recognition 
In short, you can clone tensorflow at: 
 `git clone https://github.com/tensorflow/models` 
And then find the location of the tutorial at: 
 `cd models/tutorials/image/imagenet` 
If you are on a Mac, you might need to install tensorflow with the following command: 
 `sudo -H pip install tensorflow --upgrade --ignore-installed` 
 
Usage: 
 `python active_learning_for_images.py --directory=DIRECTORY_OF_IMAGES` 
 
Where `DIRECTORY_OF_IMAGES` is the directory containing the JPGs you want apply Active Learning to. 

