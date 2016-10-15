# python assignment.py <iterations> <learning rate>

import os
import sys
import tensorflow as tf
import numpy as np
from subprocess import call

TRAIN = 'train/'
VALIDATE = 'valid/'
LABELS = 'labels.txt'
NUM_CLASSES = 104

ITERS = 1000
LEARNING_RATE = 0.5

try:
    if len(sys.argv) > 1:
        ITERS = int(sys.argv[1])
        LEARNING_RATE = float(sys.argv[2])
except:
    pass

TRAINING_DATA = 'train_preprocessed.npy'
TRAINING_LABELS = 'train_preprocessed_labels.npy'
VALIDATION_DATA = 'valid_preprocessed.npy'
VALIDATION_LABELS = 'valid_preprocessed_labels.npy'

if not os.path.isfile(TRAINING_DATA):
    call(["python", "preprocess_data.py", "TRAIN"])

if not os.path.isfile(VALIDATION_DATA):
    call(["python", "preprocess_data.py", "VALID"])

if not os.path.isfile(TRAINING_LABELS):
    call(["python", "preprocess_labels.py", "TRAIN"])

if not os.path.isfile(VALIDATION_LABELS):
    call(["python", "preprocess_labels.py", "VALID"])

x_train = np.load('train_preprocessed.npy')
y_train = np.load('train_preprocessed_labels.npy')
x_test = np.load('valid_preprocessed.npy')
y_test = np.load('valid_preprocessed_labels.npy')

print "All data loaded"

NUM_FEATURES = len(x_train[0])

x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

W = tf.Variable(tf.zeros([NUM_FEATURES, NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(
    LEARNING_RATE).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print "Starting training for " + str(ITERS)\
    + " iterations with learning rate " + str(LEARNING_RATE)

for i in range(ITERS):
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 1000))