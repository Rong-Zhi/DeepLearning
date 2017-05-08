import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data

# this net consists of 2 conv layers and 1 fully-connected layer
# conv-1
filter_size1 = 5  # kernel(filter) size
num_filters1 = 16 # number of these filters (channels)

# conv-2
filter_size2 = 5 # kernel(filter) size
num_filters2 = 36 # number of these filters (channels)

# fully-connected layer
fc_size = 128

data= input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Testing-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels,axis=1)

# Data dimension
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size,img_size)
num_channels = 1  # number of color channel---gray scale is 1 channel
num_classes = 10

# plot imgs function
def plot_images(images,cls_true,cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig,axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# test the plot function
# images = data.test.images[0:9]
# cls_true = data.test.cls[0:9]
# plot_images(images,cls_true)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    # return tf.Variable(tf.constant(0.05,shape=[length]))
    return tf.Variable(tf.truncated_normal(shape=[length],stddev=0.05))

# Helper-function for creating a single new conv layer
# input arguments: image number,
    # Y-axis of each image,
    # X-axis of each image
    # channels of each image
# output arguments: image number,
    # Y-asis of each image,--if 2*2 pooling is used, height&weight/2
    # X-axis of each image. ditto
    # channels produced by the conv filters
def new_conv_layer(input,               # the previous layer,
                   num_input_channels,  # num. channels in prev. layer
                   filter_size,         # width and height of each filter
                   num_filters,         # num. filters
                   use_pooling=True):   # use 2*2 max-pooling

    # shape of the filter-weights for convolution
    # this format is determined by the tensorflow API
    shape = [filter_size,filter_size,num_input_channels,num_filters]

    # create new weights & biases
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    # set stride to 1
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # e.g.strides = [1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')
    # Add baises to each filter-channel
    layer += biases

    if use_pooling:
        # 2x2 max-pooling, which means that we consider 2x2 windows
        # and select the largest value in each window.
        # then we move 2 pixels to the next window
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # add Rectified Linear Unit (ReLU)
    # it calculates max(x,0) for each input pixel x
    #  this ads some non-linearity to the formula and allows
    # us to learn more complicated functions
    layer = tf.nn.relu(layer)
    return layer, weights

# flatten layer:
# reduce the 4-dim tensor to 2-dim, which can be used as
# input to the fully-connected layers
def flatten_layer(layer):
    # get the shape of the input layer
    layer_shape = layer.get_shape()
    # tensor.get_shape() will return [dimension(4),dimension(3)...

    # the shape of the input layr is assumed to be:
    # layer_shape == [num_images, img_height, img_weight, num_channels]
    # the number of features is : img_height * img_weight * num_channels
    num_features = layer_shape[1:4].num_elements()

    # reshape the layer to [num_images,num_features]
    # we just se the size of the second dimensuion to num_features
    # and the size of the first dimension to -1
    # which means the size of the tensor is unchanged from reshaping
    layer_flat = tf.reshape(layer,[-1, num_features])
    # the shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features

# create a new fully-connected layer
# the input is 2-dim tensor of shape [num_images, num_inputs].
def new_fc_layer(input,         # the previous layer
                 num_inputs,    # num. inputs from prev. layer
                 num_outputs,    # num. outputs
                 use_relu=True):# use relu?
    # create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # calculate the layer as the matrix multiplication of
    # the input and weights, and then add the biases
    layer = tf.matmul(input,weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

# define placeholder
x = tf.placeholder(tf.float32, shape=[None,img_size_flat], name='x')
# -1 means keeping the shape(first dimension to None)--> number of images
# just change the img dimension
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# y_true has the size of [None, num_clasees]
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

# convolutional layer 1
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
print(layer_conv1)

# conv-layer 2
layer_conv2,weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2)
print(layer_conv2)

# flatten-layer
layer_flat, num_features = flatten_layer(layer_conv2)
print(layer_flat)

# FC layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
print(layer_fc1)

# FC layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
print(layer_fc2)

# predicted class
# use softmax to normalize them so that
# each element is limited between 0-1 and 10 elements sum to 1
y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred,dimension=1)

# use cross-entropy cost-function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=layer_fc2,labels=y_true))

# optimization method is Adamoptimizer
# (is an advanced form of gradient descent)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# performance measures
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# create tensorflow session
session = tf.Session()

# initial variables
session.run(tf.global_variables_initializer())

# define optimization function
train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
    # ensure we update the global variable rather than a local copy
    global total_iterations

    for i in range(num_iterations):

        x_batch, y_true_batch = \
            data.train.next_batch(train_batch_size)

        feed_dict_train = {x:x_batch,y_true:y_true_batch}

        session.run(optimizer,feed_dict=feed_dict_train)

        # print session every 100 iterations
        if i % 100 == 0:
            # calculate the accuracy on the training-set
            acc = session.run(accuracy,feed_dict=feed_dict_train)

            # message for printing
            msg = "Optimization Iteration:{0:>6}, Training Accuracy:{1:>6.1%}"

            print(msg.format(i,acc))

    total_iterations += num_iterations

        # define function to plot confusion matrix
def plot_confusion_matrix(cls_pred):

    cls_true = data.test.cls

    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# define function for showing the performance
test_batch_size = 256
def print_test_accuracy(show_confusion_matrix=False):
    # num. of images in the test-set
    num_test = len(data.test.images)

    cls_pred = np.zeros(shape=num_test, dtype=int)

    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        # get the imgs from test-set between index i-j
        images = data.test.images[i:j,:]

        # associated labels
        labels = data.test.labels[i:j,:]

        feed_dict = {x:images,y_true:labels}

        cls_pred[i:j] = session.run(y_pred_cls,feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    print("Accuracy on Test-set:{0:.1%}({1}/{2})".format(
        acc,correct_sum,num_test))

    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# performance after 100 optimization iteration
optimize(num_iterations=500)
print_test_accuracy(show_confusion_matrix=True)


##########
# output##
##########
# Extracting data/MNIST/train-images-idx3-ubyte.gz
# Extracting data/MNIST/train-labels-idx1-ubyte.gz
# Extracting data/MNIST/t10k-images-idx3-ubyte.gz
# Extracting data/MNIST/t10k-labels-idx1-ubyte.gz
# Size of:
# - Training-set:		55000
# - Testing-set:		10000
# - Validation-set:	5000
# Tensor("Relu:0", shape=(?, 14, 14, 16), dtype=float32)
# Tensor("Relu_1:0", shape=(?, 7, 7, 36), dtype=float32)
# Tensor("Reshape_1:0", shape=(?, 1764), dtype=float32)
# Tensor("Relu_2:0", shape=(?, 128), dtype=float32)
# Tensor("add_3:0", shape=(?, 10), dtype=float32)
# 2017-05-08 14:11:41.190423: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-08 14:11:41.190444: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-08 14:11:41.190449: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# Optimization Iteration:     0, Training Accuracy:  7.8%
# Optimization Iteration:   100, Training Accuracy: 65.6%
# Optimization Iteration:   200, Training Accuracy: 79.7%
# Optimization Iteration:   300, Training Accuracy: 90.6%
# Optimization Iteration:   400, Training Accuracy: 87.5%
# Accuracy on Test-set:91.1%(9114/10000)
# Confusion matrix:
# [[ 939    0    3    7    0    6   12    1   12    0]
#  [   0 1092    3    8    0    1    6    0   25    0]
#  [  12    1  894   24   12    1   26   18   42    2]
#  [   2    3   14  937    0   20    1   11   17    5]
#  [   1    2    5    0  887    1   24    3    6   53]
#  [   3    1    2   61   15  767   19    1   19    4]
#  [   9    5    3    2    5   19  909    0    6    0]
#  [   1    9   27    9    5    1    0  910    7   59]
#  [   3    2    8   34   12   13    8   11  876    7]
#  [   9    5    7   16   28    7    0   26    8  903]]