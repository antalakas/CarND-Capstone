#!/usr/bin/env python

import tensorflow as tf
import keras

from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

X_valid = X_test
y_valid = y_test

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np
import random
import pandas


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

n_valid = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = [X_train.shape[1],X_train.shape[2],X_train.shape[3]]

# TODO: How many unique classes/labels there are in the dataset.
def get_cnt(lVals):
    d = dict(zip(lVals, [0] * len(lVals)))
    for x in lVals:
        d[x] += 1
    return d
signsDict = get_cnt(y_train)
n_classes = len(signsDict)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


def getNsigns(y_values):
    signsCountInternal = np.zeros(n_classes)
    for i in range(y_values.shape[0]):
        signsCountInternal[y_values[i]] = signsCountInternal[y_values[i]] + 1
    return signsCountInternal
    
signsCount = getNsigns(y_train)
print(signsCount)


### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


grayconversion_train = tf.image.rgb_to_grayscale(X_train)
with tf.Session() as sess:
    grayscale_train = sess.run(grayconversion_train)
grayconversion_valid = tf.image.rgb_to_grayscale(X_valid)
with tf.Session() as sess:
    grayscale_valid = sess.run(grayconversion_valid)
grayconversion_test = tf.image.rgb_to_grayscale(X_test)
with tf.Session() as sess:
    grayscale_test = sess.run(grayconversion_test)
print(grayscale_train.shape)
print(grayscale_valid.shape)
print(grayscale_test.shape)


X_train_all = np.append(X_train,grayscale_train,axis=3)
X_test_all = np.append(X_test,grayscale_test,axis=3)
X_valid_all = np.append(X_valid,grayscale_valid,axis=3)
print(X_train_all.shape)
print(X_valid_all.shape)
print(X_test_all.shape)

### Define your architecture here.
### Feel free to use as many code cells as needed.

EPOCHS = 50
BATCH_SIZE = 256

layer1 = (3,3,3,24)
layer2 = (3,3,24,32)
layer3 = (3,3,32,64)
layer1g = (3,3,1,24)
layer2g = (3,3,24,32)
layer3g = (3,3,32,64)
layer4 = (2816,352)
layer5 = (352,176)
layer6 = (176,n_classes)

from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.05

conv1_W = tf.Variable(tf.truncated_normal(shape=layer1, mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(layer1[3]))

conv2_W = tf.Variable(tf.truncated_normal(shape=layer2, mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(layer2[3]))

conv3_W = tf.Variable(tf.truncated_normal(shape=layer3, mean = mu, stddev = sigma))
conv3_b = tf.Variable(tf.zeros(layer3[3]))

conv1_Wg = tf.Variable(tf.truncated_normal(shape=layer1g, mean = mu, stddev = sigma))
conv1_bg = tf.Variable(tf.zeros(layer1g[3]))

conv2_Wg = tf.Variable(tf.truncated_normal(shape=layer2g, mean = mu, stddev = sigma))
conv2_bg = tf.Variable(tf.zeros(layer2g[3]))

conv3_Wg = tf.Variable(tf.truncated_normal(shape=layer3g, mean = mu, stddev = sigma))
conv3_bg = tf.Variable(tf.zeros(layer3g[3]))

fc1_W = tf.Variable(tf.truncated_normal(shape=layer4, mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(layer4[1]))

fc2_W  = tf.Variable(tf.truncated_normal(shape=layer5, mean = mu, stddev = sigma))
fc2_b  = tf.Variable(tf.zeros(layer5[1]))

fc3_W  = tf.Variable(tf.truncated_normal(shape=layer6, mean = mu, stddev = sigma))
fc3_b  = tf.Variable(tf.zeros(n_classes))

keep_prob = tf.placeholder("float")

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    x_color = x[:,:,:,0:3]
    x_gray = x[:,:,:,2:3]
    conv1 = tf.nn.conv2d(x_color, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv1g = tf.nn.conv2d(x_gray, conv1_Wg, strides=[1, 1, 1, 1], padding='VALID') + conv1_bg
    conv1g = tf.nn.relu(conv1g)
    conv1g = tf.nn.max_pool(conv1g, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2g = tf.nn.conv2d(conv1g, conv2_Wg, strides=[1, 1, 1, 1], padding='VALID') + conv2_bg
    conv2g = tf.nn.relu(conv2g)
    conv2g = tf.nn.max_pool(conv2g, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3g = tf.nn.conv2d(conv2g, conv3_Wg, strides=[1, 1, 1, 1], padding='VALID') + conv3_bg
    conv3g = tf.nn.relu(conv3g)
    conv3g = tf.nn.max_pool(conv3g, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    fc0   = tf.concat([flatten(conv3),flatten(conv2),flatten(conv3g),flatten(conv2g)],1)
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1   = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
    
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 4))
y = tf.placeholder(tf.int32, (None))
print(y)
one_hot_y = tf.one_hot(y, n_classes)
print(y)


rate = 0.001
beta = 0.001

logits = LeNet(x)
print(x)
print(one_hot_y)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy + \
    beta*tf.nn.l2_loss(conv1_W) + \
    beta*tf.nn.l2_loss(conv2_W) + \
    beta*tf.nn.l2_loss(conv3_W) + \
    beta*tf.nn.l2_loss(fc1_W) + \
    beta*tf.nn.l2_loss(fc2_W))
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
    
    
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_all)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_all, y_train = shuffle(X_train_all, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_all[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
            
        training_accuracy = evaluate(X_train_all, y_train)
        validation_accuracy = evaluate(X_valid_all, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test_all, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
