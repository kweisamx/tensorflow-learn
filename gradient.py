import tensorflow as tf 
import numpy as np


# Create data

x_data = np.random.rand(100).astype(np.float) # random create the 100 number

y_data = x_data*0.1 + 0.3 ## the true function


''' create tensorflow structure start'''
# We know the w = 0.1 and b = 0.3 ,
# but we want to that tf to figure out the number
Weights = tf.Variable(tf.random_uniform([1], -100.0 , 100.0 )) # create 1 dimension , range in (-1,1) ,tf.Variable is a variable
biases = tf.Variable(tf.random_uniform([1],-10.0,10.0)) #Biases is zero


y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) # loss function

optimizer = tf.train.GradientDescentOptimizer(0.5) # set learn rate 

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

''' create tensorflow structure end '''

sess = tf.Session()
##
print("=====================================")
print("data is ",x_data)

sess.run(init)


for step in range(401):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights),sess.run(biases))

