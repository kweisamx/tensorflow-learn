import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./",one_hot=True)

# The mnist_data.read_data_sets will automatically split data set to three part,test ,train ,validation


# print the Train data size: 55000
print( "Training data size:", mnist.train.num_examples)

# print validating data size: 5000
print( "validating data size:",mnist.validation.num_examples)

# print Testing data size: 1000
print( "Testing data size:", mnist.test.num_examples)

# print Example training data :[0. 0. 0. ... 0.38 0.376 .... 0.]
# The range [0,1] means the color depth, 0 is white , 1 is the dark
print("Example training data:", mnist.train.images[0])

# print Example training data label:
# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Example training data label: ", mnist.train.labels[0])


# MNIST data set constant

INPUT_NODE = 784 # The input layer node  number, for MNIST dataset, it's like the graph pixel
OUTPUT_NODE = 10 # Output layer node number ,because in MNIST dataset the number is [0,9], so the output layer number is 10



LAYER1_NODE = 500 # hiden layer node number , here we just use one hiden layer to do example

BATCH_SIZE = 100 # one batch train dataset number, if number is become samll , the train close random gradient desent . number is become bigger , the train close to gradient desent


LEARNING_RATE_BASE = 0.8 # The basic learning 
LEARNIING_RATE_DECAY = 0.99 # The  desent of learning rate  
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

