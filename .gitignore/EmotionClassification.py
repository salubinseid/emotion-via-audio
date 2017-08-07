
import sys
import time
import os
import glob
import numpy
import cPickle
import aifc
import math
import itertools
from sklearn.metrics import precision_recall_fscore_support
from numpy import NaN, Inf, arange, isscalar, array as np 
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.signal import lfilter, hamming

import librosa
import librosa.display
import tensorflow as tf
from matplotlib.pyplot import specgram 

from sklearn.metrics import confusion_matrix

# user defined liberary
from featureExtraction import *  

#Configuarion parameters for the neural networks 
########################################################################
training_epochs = 24000 # 50
n_dim = tr_features.shape[1]  # 193 features are inputed to the NN

n_classes = 8 # old 10
n_hidden_units_one = 280 
n_hidden_units_two = 300


sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01
#########################################################################

########################################################################################################
# define placeholders for features and class labels, which tensor flow will fill with the data at runtime.
#
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])
#########################################################################################################

# #######################################################################################################
# define weights and biases for hidden and output layers of the network. For non-linearity, 
# we use the sigmoid function in the first hidden layer and tanh in the second hidden layer. 
#
W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd, name='w1'))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd, name='bias'))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd, name='w2'))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)
'''
W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)
'''
###################################################################################################

########################################################################################################
# The output layer has softmax as non-linearity as we are dealing with multiclass classification problem. 
#
W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b, name='output')
###################################################################################################

init = tf.initialize_all_variables()

#cost_function = -tf.reduce_sum(Y * tf.log(y_))
cost_function = -tf.reduce_mean(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Optimizer 
cost_history = np.empty(shape=[1],dtype=float)

y_true, y_pred = None, None

# Create a saver.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):            
        _, cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)

        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=1000)

        print "Epoch:", epoch, "Cost History:", cost 
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))
    
    print("Test accuracy: ",round(sess.run(accuracy,feed_dict={X: ts_features,Y: ts_labels}),3))


    #--------------------


def prediction():
	print "the prediction will goes here"



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
class_names = ["angry","exc","fea","fru","hap","neu","sad","sur"]
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
print cnf_matrix
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()


fig = plt.figure(figsize=(10,8))

plt.plot(cost_history)

plt.ylabel("Cost")
plt.xlabel("Iterations / training_epochs")

plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print "F-Score:", round(f,3)