import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# confusion matrix is a table that is often used to describe
# the performance of a classification model
# ##########################################
#  C00 true negative  # C01 false positive
#  C10 false negative # C11 true positive
# ##########################################

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# print(confusion_matrix(y_true,y_pred))

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/",one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# Data dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes =10

batch_size = 100 # SGD batch size


data.test.cls = np.array([label.argmax() for label in data.test.labels])
# plot images
def plot_images(images, cls_true,cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3*3 sub-plots
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
# images = data.test.images[0:9]
#
# cls_true = data.test.cls[0:9]
#
# plot_images(images=images,cls_true=cls_true)
#
# plt.show()

# define placeholder
#  x has shape [num_images, img_size_flat] and
x = tf.placeholder(tf.float32, [None,img_size_flat])
y_true = tf.placeholder(tf.float32,[None,num_classes])
y_true_cls = tf.placeholder(tf.int64,[None])

# define variables
# weights has shape [img_size_flat, num_classes]
weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# define model
# The result is a matrix of shape [num_images, num_classes]
logits = tf.add(tf.matmul(x,weights),biases)

# each row sums to 1, every element is limited between 0-1
y_pred = tf.nn.softmax(logits)
# the predicted class can be calculated from the y_pred matrix
#  by taking the index of the largest element in each row
y_pred_cls = tf.argmax(y_pred, dimension=1)

# The cross-entropy is a continuous function that
# is always positive and if the predicted output of
# the model exactly matches the desired output then
# the cross-entropy equals zero.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)

# the average of cross_entropy is defined as loss(a single scalar value)
cost = tf.reduce_mean(cross_entropy)

# define the optimization way---gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

# optimization function
def optimization(num_iterations):
    for i in range(num_iterations):
        x_batch,y_trian_batch = data.train.next_batch(batch_size)

        feed_dict_train = {x:x_batch,y_true:y_trian_batch}
        session.run(optimizer,feed_dict=feed_dict_train)

feed_dict_test = {x:data.test.images,
                  y_true:data.test.labels,
                  y_true_cls:data.test.cls}
#
# print accuracy
def print_accuracy():
    acc = session.run(accuracy,feed_dict=feed_dict_test)
    print("Accuracy on test-set:{0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true = data.test.cls

    cls_pred = session.run(y_pred_cls,feed_dict=feed_dict_test)

    # define confusion matrix
    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)

    print(cm)

    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_exmaple_errors():

    correct, cls_pred = session.run([correct_prediction,y_pred_cls],
                                    feed_dict=feed_dict_test)
    incorrect = (correct == False) # negate the boolean array

    images = data.test.images[incorrect]

    # get the predicted classes for those incorrect imgs
    cls_pred = cls_pred[incorrect]

    # get the true classes for those incorrect imgs
    cls_true = data.test.cls[incorrect]

    # plot the first 9 imgs
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    plt.show()
#####performance before any optimization
# print_accuracy()
# plot_exmaple_errors()
# plt.show()

#####performance after one optimization
# optimization(num_iterations = 1)
# print_accuracy()
# plot_exmaple_errors()
# plt.show()

####performance after 100 optimization
optimization(num_iterations = 100)
print_accuracy()
plot_exmaple_errors()
print_confusion_matrix()




