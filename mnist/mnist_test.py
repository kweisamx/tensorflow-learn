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


# Forward

def inference(input_tensor,avg_class, weights1, biases1,weights2, biases2):
    if avg_class ==None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+ biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + 
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# Training 

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE],name = 'y-input')

    # Create hidden layer function parameter
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # Create output layer function parameter
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # Now calculate the forward propagation
    y = inference(x, None , weights1,biases1,weights2,biases2)
    
    global_step = tf.Variable(0,trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)


    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())

    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2)


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
    .minimize(loss, global_step = global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()


        validate_feed = {x: mnist.validation.images,
                        y_:mnist.validation.labels}

        test_feed = {x:mnist.test.images, y_:mnist.test.labels}


        for i in range(TRANING_STEPS):
            if i% 1000 == 0:
                xs,ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op,feed_dict={x:xs,y:ys})


        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training stets, test accuracy using averages")

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
