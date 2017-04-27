import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X,w):
    return tf.matmul(X,w)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=true)
