#   Simple Deep Feedforward Neural Network(Multilayer Perceptron)
#   Tensorflow tutorial mnist dataset
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X,w_h,w_o):
    h = tf.nn.sigmoid(tf.matmul(X,w_h))
    return tf.matmul(h,w_o)
