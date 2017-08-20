
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflw.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity.tf.logging.INFO

# Our application logic will be added here

if __name__=="__main__":
    tf.app.run()


# In[2]:


def cnn_model_fn(features, labels ,mode):
    """Model function for CNN"""
    
    #Input Layer
    input_layer = tf.reshape(features,[-1,28,28,1])# Batch_size?,image_width,image_height,channel
    # 1*28*28
    # Convolutional Layer #1
    conv1 = tf.laters.conv2d(
        inputs=input_layer, #shape = [batch_size e, 28,28,1]
        filters=32,
        kernel_size=[5,5], # [5,5]==>5
        padding="same",# output tensor preserve width and height 20
        activation=tf.nn.relu)
    # Polling Layer #1
    # 32*24*24
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides = 2)
    
    # Convolution Layer #2 and Pooling Layer #2
    conv2 = tf.laters.conv2d(
        inputs=pools1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides = 2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learm.ModeKeys.TRAIN)
    
    # Ligits Layer
    logits = tf.layers.dense(inputs=fropout, units=10)
    
    loss = None
    train_op = None
    
    # Calculate Loss(for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
        loss = tf.loss.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        
        
    # Calculate the Training Op (for TRAIN mode)
    if mode ==learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate = 0.001,
                optimizer="SGD")
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits,axis = 1),
        "probabilities":tf.nn.softmax(
            logits, name="softmax_tensor")
        
    }
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss = loss , train_op = train_op)


# In[ ]:




