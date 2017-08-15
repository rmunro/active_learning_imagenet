# Active Learning with TensorFlow and ImageNet
 Coding exercise to extend TensorFlow for ImageNet to Active Learning, for use in a job interview or similar context. 
 
The code runs, but _IS DELIBERATELY A BAD IMPLEMENTATION OF ACTIVE LEARNING_ 

The task is to fix this code.

# Context 

A company has approached you that wants to classify a large set of sports images according to the ImageNet set of labels. 

However, there are several problems:
 1. They are trying to classify with the pre-compiled TensorFlow model, but the model is not very accurate on their data. So, they want to manually label some of their images to retrain the classifier on their own images.
 1. They have millions of images, so they cannot manually label them all.
 1. Their collection of images may include some that are not related to sports. They want to make sure the sports images are filtered out, but they don't otherwise care about these images.

This is a real-world situation that occurs regularly. For this exercise, we will use the open set of sports images from CrowdFlower's Data for Everyone program:

 https://www.crowdflower.com/data-for-everyone/ 

## About ImageNet 

ImageNet uses a classification scheme based on WordNet, where words are grouped by synonyms, called 'synsets'. Each 'synset' is a group of closely related words. These synsets are the labels for this task, which you will see when you run the code. For example, the label for 'racing car' is `'racer, race car, racing car'`.

WordNet organizes the synsets in hierarchies. For example 'baseball' and 'cricket' could be types of 'sport', and in turn 'sport' could be a type of 'activity'. Generally, items that are closer in the hierarchy tend to be closer in real-life. For example, 'sports car' and 'racing car' are both types of 'cars' in WordNet/ImageNet, and are also closely related in real-life. By contrast, 'sports car' and 'pine tree' are not closely related in WordNet/ImageNet, or in real-life.

 
## Getting started 

A (250MB) subset of the images is available at:

http://www.robertmunro.com/research/test_images.tar 

The starter code is in the same directory as this readme:

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
 
There are many extensions to this code, from a 1 hour exercise to improve how confidence is used to order the images, to a multiple week exercice that could included retraining all/parts of the model and providing interfaces that are optimal for different kinds of human labeling.

## 2 Hour Exercise

For the 2 Hour Exercise, there is a coding and a written component. 

It is recommended that you take 30 minutes to become familiar with the problem and decide on your approach, 60 minutes for the coding exercise, and 30 minutes on the writing exercise.

You may use any resources that are available to you on your machine or on the internet. You can ask the instructor for any clarifications questions, but please complete this as a solo exercise without live input from other people.

### Coding Exercise

Reimplement `order_images_for_active_learning()` so that it has a better strategy to determine which images will be the most valuable to classify. Consider the following:
 1. Select images that are hard for the current TensorFlow model to classify
 1. Select images that seem unknown to the current TensorFlow model
 1. Select images that cover as broad a selection of labels as possible

### Written Exercise

Write a 1-paragraph description of what you implemented in the coding exercise, justifying each decision. There are many possible solutions that can be implemented in about 60 minutes, so the evaluation is more about your reasoning than your exact strategy.

Please also write a few paragraphs or bullet points covering other strategies that you might try if you had more time, covering: 
 1. How else could you ensure that you covered as broad a selection of images as possible?
 1. What kind of user interfaces could a human annotator use to manually add the labels efficiently and accurately? 
 1. How would you retrain the model with the newly labelled images, and what might be the pros and cons of retraining TensorFlow on the newly labeled images very frequently vs training only a few times or even just once?
 1. How would you evaluate the effectiveness of your strategies to see what worked best?

### Submitting the Exercise

Please email the updated code and written responses to the instructor when you are complete.
 

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

And then find the location of the tutorial from which this was based at: 

 `cd models/tutorials/image/imagenet` 

This tutorial is not required reading for this exercise, but will give you more context if you are not familiar with TensorFlow or ImageNet.

If you are on a Mac, you might need to install TensorFlow with the following command: 

 `sudo -H pip install tensorflow --upgrade --ignore-installed` 
 
Usage: 

 `python active_learning_for_images.py --directory=DIRECTORY_OF_IMAGES` 
 
Where `DIRECTORY_OF_IMAGES` is the directory containing the JPGs you want apply Active Learning to. 

