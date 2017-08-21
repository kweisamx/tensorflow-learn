
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np

# define add one layer function
def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors,output_tensors]))
    b = tf.Variable(tf.zeros([1, output_tensors]))
    formula = tf.add(tf.matmul(inputs,w),b)
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_fuction(formula)
    return outputs
x_data = np.random.rand(100)
x_data = x_data.reshape(len(x_data),1)
y_data = x_data * 0.1 + 0.3

# build Feeds
x_feeds = tf.placeholder(tf.float32, shape = [None,1])
y_feeds = tf.placeholder(tf.float32, shape = [None,1])

# add one hidden layer
hidden_layer = add_layer(x_feeds, input_tensors = 1, output_tensors = 10, activation_function = None)

# add one output layer
output_layer =add_layer(hidden_layer, input_tensors = 10, output_tensors = 1, activation_function = None)

# define loss function

loss = tf.reduce_mean(tf.square(y_feeds - output_layer))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# init the graph and compute

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train, feed_dict = {x_feeds:x_data,y_feeds:y_data})
        if step % 20 ==0:
            print(sess.run(loss,feed_dict = {x_feeds:x_data,y_feeds: y_data}))
    


# In[ ]:




