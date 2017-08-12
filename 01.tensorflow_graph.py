import tensorflow as tf 


# at graph1 we define the variable "v", and set it zeros
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        "v",initializer=tf.zeros_initializer(shape=[1]))

# at graph1 we define the variable "v", and set it zeros
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        "v",initializer=tf.ones_initializer(shape=[1]))

#get the graph1 variable v
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))


#get the graph2 variable v
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))
