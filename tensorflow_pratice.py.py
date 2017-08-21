
# coding: utf-8

# In[9]:


#mul matrix
import tensorflow as tf
#1*2Matrix
matrix1 = tf.constant([[3,3]])
#2*1Matrix 
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run([product])
    print(result)


# In[20]:


# use operator for constant
state = tf.Variable(0,name="counter")
one = tf.constant(1)
new_value  = tf.add(state, one)
update = tf.assign(state, new_value)
## initializer
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    # update 3 times and print
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# In[19]:


# use placeholder to input data
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2) # multiply is new function for mul

with tf.Session() as sess:
    print(sess.run([output], feed_dict = {input1:[7],input2:[3]}))


# In[22]:


# out data 
input1 = tf.constant([3])
input2 = tf.constant([5])
added = tf.add(input1, input2)
multiplied = tf.multiply(input1,input2)

with tf.Session() as sess:
    result = sess.run([added, multiplied])
    print(result)


# In[ ]:




