#!/usr/bin/env python

import glob
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import numpy as np
from sklearn.utils import shuffle


preprocess_images = False

with tf.Session():
	counter = 0
	if preprocess_images == True:
		
		n_im = glob.glob('tl_classifier_exceptsmall/simulator/NoTrafficLight/*.png')
		y_im = glob.glob('tl_classifier_exceptsmall/simulator/Yellow/*.png')
		g_im = glob.glob('tl_classifier_exceptsmall/simulator/Green/*.png')
		r_im = glob.glob('tl_classifier_exceptsmall/simulator/Red/*.png')
		
		#print(n_im)
		#print(y_im)
		#print(g_im)
		#print(r_im)
		
		for temp_im in n_im:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/n_im//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in y_im:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/y_im//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in g_im:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/g_im//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in r_im:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/r_im//'+str(counter)+'.png',resized_image)
			counter+=1
			
with tf.Session():
	counter = 0
	if preprocess_images == True:
		
		n_im_real = glob.glob('tl_classifier_exceptsmall/real/NoTrafficLight/*.png')
		y_im_real = glob.glob('tl_classifier_exceptsmall/real/Yellow/*.png')
		g_im_real = glob.glob('tl_classifier_exceptsmall/real/Green/*.png')
		r_im_real = glob.glob('tl_classifier_exceptsmall/real/Red/*.png')
		
		#print(n_im)
		#print(y_im)
		#print(g_im)
		#print(r_im)
		
		for temp_im in n_im_real:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/n_im_real//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in y_im_real:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/y_im_real//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in g_im_real:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/g_im_real//'+str(counter)+'.png',resized_image)
			counter+=1
			
		for temp_im in r_im_real:
			newimage = cv2.imread(temp_im)
			resized_image = tf.image.resize_image_with_crop_or_pad(newimage, 150, 50)
			resized_image = resized_image.eval()
			cv2.imwrite('processed/r_im_real//'+str(counter)+'.png',resized_image)
			counter+=1

n_im = glob.glob('processed/n_im/*.png')
y_im = glob.glob('processed/y_im/*.png')
g_im = glob.glob('processed/g_im/*.png')
r_im = glob.glob('processed/r_im/*.png')

n_im_real = glob.glob('processed/n_im_real/*.png')
y_im_real = glob.glob('processed/y_im_real/*.png')
g_im_real = glob.glob('processed/g_im_real/*.png')
r_im_real = glob.glob('processed/r_im_real/*.png')

NVars = 150*50*3
n_classes = 4

X_train = np.array([])
Y_train = np.array([])

for temp_im in n_im:
	newimage = cv2.imread(temp_im)
	#print(X_train.shape)
	#print(np.array([newimage]).shape)
	#newimage = np.reshape(newimage,-1)
	if X_train.shape[0] == 0:
		X_train = np.array([newimage])
	else:	
		X_train = np.concatenate((X_train, np.array([newimage])),axis=0)	
	Y_train = np.append(Y_train, 0)
	
for temp_im in y_im:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_train = np.concatenate((X_train,np.array([newimage])),axis=0)	
	Y_train = np.append(Y_train, 1)
	
for temp_im in g_im:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_train = np.concatenate((X_train,np.array([newimage])),axis=0)	
	Y_train = np.append(Y_train, 2)
	
for temp_im in r_im:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_train = np.concatenate((X_train,np.array([newimage])),axis=0)	
	Y_train = np.append(Y_train, 3)

X_train, Y_train = shuffle(X_train, Y_train)

print("Train shape: ")
print(X_train.shape)
#print(Y_train)

X_test = np.array([])
Y_test = np.array([])

for temp_im in n_im_real:
	newimage = cv2.imread(temp_im)
	#print(X_train.shape)
	#print(np.array([newimage]).shape)
	#newimage = np.reshape(newimage,-1)
	if X_test.shape[0] == 0:
		X_test = np.array([newimage])
	else:	
		X_test = np.concatenate((X_test, np.array([newimage])),axis=0)	
	Y_test = np.append(Y_test, 0)
	
for temp_im in y_im_real:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_test = np.concatenate((X_test,np.array([newimage])),axis=0)	
	Y_test = np.append(Y_test, 1)
	
for temp_im in g_im_real:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_test = np.concatenate((X_test,np.array([newimage])),axis=0)	
	Y_test = np.append(Y_test, 2)
	
for temp_im in r_im_real:
	newimage = cv2.imread(temp_im)
	#newimage = np.reshape(newimage,-1)
	X_test = np.concatenate((X_test,np.array([newimage])),axis=0)	
	Y_test = np.append(Y_test, 3)

X_test, Y_test = shuffle(X_test, Y_test)

print("Test shape: ")
print(X_test.shape)

EPOCHS = 7
BATCH_SIZE = 16

layer1 = (5,5,3,4)
layer2 = (5,5,4,6)
layer3 = (5,5,6,7)
layer4 = (2046,25)
layer5 = (25,10)
layer6 = (10,n_classes)

from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.05

conv1_W = tf.Variable(tf.truncated_normal(shape=layer1, mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(layer1[3]))

conv2_W = tf.Variable(tf.truncated_normal(shape=layer2, mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(layer2[3]))

conv3_W = tf.Variable(tf.truncated_normal(shape=layer3, mean = mu, stddev = sigma))
conv3_b = tf.Variable(tf.zeros(layer3[3]))

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
    
    fc0   = tf.concat([flatten(conv3),flatten(conv2)],1)
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1   = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
    
x = tf.placeholder(tf.float32, (None, 150, 50, 3))
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
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
            
        training_accuracy = evaluate(X_train, Y_train)
        #validation_accuracy = evaluate(X_valid_all, Y_valid)
        validation_accuracy = evaluate(X_train, Y_train)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

def predict_test(X_data, y_data):
    sess = tf.get_default_session()
    prediction = sess.run(tf.argmax(logits,1), feed_dict={x: X_data, y: y_data ,keep_prob : 1.0})
    prediction = sess.run(tf.nn.top_k(logits,k=1), feed_dict={x: X_data, y: y_data ,keep_prob : 1.0})
    return prediction
    
    
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint('.'))
	predictions = predict_test(X_test, Y_test)
	indices = predictions.indices
	#print(predictions.indices)

incorrect = 0
for i in range(Y_test.shape[0]):
	if(Y_test[i] != indices[i][0]):
		incorrect += 1
		#print(i)
	#print(Y_test[i] , indices[i][0])

print(incorrect)





