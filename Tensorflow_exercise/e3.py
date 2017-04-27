#   Logistic Regression
#   Tensorflow tutorials mnist dataset

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X,w):
    return tf.matmul(X,w)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#   create symbolic variables
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w = init_weights([784,10])

py_x = model(X,w)

#   cross entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#   GD optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
#  evaluate the argmax of the logistic regression during prediction
predict_op = tf.argmax(py_x,1)

# now launch the session
with tf.Session() as sess:
    # initialization
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1), 128):
            sess.run(train_op, feed_dict=={X:trX[start:end], Y:trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X,teX})))
