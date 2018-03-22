import numpy as np
import sys
import re
import codecs
import os
import jieba
from gensim.models import word2vec
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import gensim
from sklearn.linear_model import LogisticRegression    
from random import shuffle
from sklearn.svm import SVC    
from sklearn.naive_bayes import MultinomialNB    
from sklearn.neighbors import KNeighborsClassifier    



def getWordVecs(wordList):
	vecs = []
	sentence_vec=[]
	count=0
	for word in wordList:
		word = word.replace('\n', '')
		try:
			word_vec=model[word].tolist()
			sentence_vec.append(word_vec)
			vecs.append(model[word])
			count=count+1
		except KeyError:
			continue
	trainset.append(sentence_vec)
	trainset_seqlen.append(count)
	return np.array(vecs, dtype = 'float')


def buildVecs(filename):
	count=0;
	posInput = []
	with open(filename, "rb") as txtfile:
		# print txtfile
		for lines in txtfile:
			lines = lines.split('\n ')
			for line in lines:
				line = jieba.cut(line)
				resultList = getWordVecs(line)
				# for each sentence, the mean vector of all its vectors is used to represent this sentence
				if len(resultList) != 0:
					resultArray = sum(np.array(resultList))/len(resultList)
					new=resultArray
					posInput.append(resultArray)
				else:
					posInput.append(new)	
	return posInput

#stop_words=[]
#f=open('stop.txt')
#while 1:
	#line=f.readline()
	#line=line.strip()
	#line=line.replace(' ','')
	#stop_words.append(line)
	#if not line:
		#break;

#sentiment_words=[]
#f=open('sentiment.txt')
#while 1:
	#line=f.readline()
	#line=line.strip()
	#line=line.replace(' ','')
	#sentiment_words.append(line)
	#if not line:
		#break;

# load word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format("total_vectors.bin", binary = True)

trainset=[]
trainset_seqlen=[]

posInput = buildVecs('total_pos.txt')
negInput = buildVecs('total_neg.txt')

shuffle(posInput) 
shuffle(negInput) 

train_X=posInput[3000:]+negInput[3000:]
train_Y=np.concatenate((np.ones(12000),np.zeros(12000)))

test_X=posInput[:3000]+negInput[:3000]
test_Y=np.concatenate((np.ones(3000),np.zeros(3000)))


#Logistic Regression
model1 = LogisticRegression(penalty='l2')    
model1.fit(train_X,train_Y) 
print 'Logistic Regression Test Accuracy: %.5f'% model1.score(test_X,test_Y)


#KNN
model2 = KNeighborsClassifier()    
model2.fit(train_X, train_Y)
print 'KNN Test Accuracy: %.5f'% model2.score(test_X,test_Y)



#CNN
import tensorflow as tf
import random

# Parameters
learning_rate = 0.0005
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 200 
n_classes = 2 
dropout = 0.75 

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

a=[1.0,0.0]
test_Y=[]
b=[0.0,1.0]
for i in range(3000):
	test_Y.append(a)
for j in range(3000,6000):
	test_Y.append(b)

train_Y=[]
for z in range(12000):
	train_Y.append(a)
for w in range(12000,24000):
	train_Y.append(b)

c=[]
train=[]
for m in range(24000):
	c=train_X[m].tolist()+train_Y[m]
	train.append(c)

for i in range(1000):
	shuffle(train)

train_X=[]
train_Y=[]
for q in range(24000):
	qq=train[q]
	qw=qq[:200]
	qz=qq[200:]
	train_X.append(qw)
	train_Y.append(qz)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
	                      padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 20,10, 1])

	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)

	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)

	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Store layers weight & bias

h1=10
w1=5
c1=32
h2=10
w2=5
c2=64
nh=1024
weights = {
   
        'wc1': tf.Variable(tf.random_normal([h1, w1, 1, c1])),
       
        'wc2': tf.Variable(tf.random_normal([h2, w2, c1, c2])),
        
        'wd1': tf.Variable(tf.random_normal([5*3*c2, nh])),
        
        'out': tf.Variable(tf.random_normal([nh, n_classes]))
}

biases = {
        'bc1': tf.Variable(tf.random_normal([c1])),
        'bc2': tf.Variable(tf.random_normal([c2])),
        'bd1': tf.Variable(tf.random_normal([nh])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		listt=random.sample(range(0, 24000), batch_size)
		batch_x=[]
		batch_y=[]
		for iii in range(len(listt)):
			batch_x.append(train_X[listt[iii]])
			batch_y.append(train_Y[listt[iii]])
		batch_x=np.array(batch_x)
		batch_y=np.array(batch_y).reshape(batch_size,2)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
		                               keep_prob: dropout})
		if step % display_step == 0:
			# Calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
			                                                  y: batch_y,
			                                                  keep_prob: 1.})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
			      "{:.6f}".format(loss) + ", Training Accuracy= " + \
			      "{:.5f}".format(acc))
		step += 1
	print("Optimization Finished!")

	test_X=np.array(test_X)
	test_Y=np.array(test_Y).reshape(6000,2)
	print("CNN Testing Accuracy:", \
	      sess.run(accuracy, feed_dict={x: test_X,
	                                    y: test_Y,
	                                    keep_prob: 1.}))




#Dynamic Recurrent Neural Networks
import tensorflow as tf
import random
# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
seq_max_len = 200 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

add=[]
for i in range(200):
	add.append(0.0)


for i in range(len(trainset)):
	for j in range(seq_max_len-trainset_seqlen[i]):
		trainset[i].append(add)


for i in range(len(trainset)):
	if trainset_seqlen[i]==0:
		trainset_seqlen[i]=1

trainset_y=[]
aa=[1.0,0.0]
bb=[0.0,1.0]
for ii in range(15000):
	trainset_y.append(aa)
for jj in range(15000,30000):
	trainset_y.append(bb)

train_stat=[]
for m in range(3000,15000):
	cc=[]
	cc.append(trainset_seqlen[m])
	c=trainset[m]+trainset_y[m]+cc
	train_stat.append(c)
for m in range(18000,30000):
	cc=[]
	cc.append(trainset_seqlen[m])
	c=trainset[m]+trainset_y[m]+cc
	train_stat.append(c)	

for i in range(1000):
	shuffle(train_stat)

trainset=[]
trainset_y=[]
trainset_seqlen=[]
#print train_stat[0]
for q in range(24000):
	qq=train_stat[q]
	qw=qq[:200]
	qz=qq[200:202]
	qt=qq[202]
	trainset.append(qw)
	trainset_y.append(qz)
	trainset_seqlen.append(qt)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 200])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Permuting batch_size and n_steps
	x = tf.transpose(x, [1, 0, 2])
	# Reshaping to (n_steps*batch_size, n_input)
	x = tf.reshape(x, [-1, 200])
	# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.split(0, seq_max_len, x)

	# Define a lstm cell with tensorflow
	lstm_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

	# Get lstm cell output, providing 'sequence_length' will perform dynamic
	# calculation.
	outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
	                            sequence_length=seqlen)

	# When performing dynamic calculation, we must retrieve the last
	# dynamically computed output, i.e., if a sequence length is 10, we need
	# to retrieve the 10th output.
	# However TensorFlow doesn't support advanced indexing yet, so we build
	# a custom op that for each sample in batch size, get its length and
	# get the corresponding relevant output.

	# 'outputs' is a list of output at every timestep, we pack them in a Tensor
	# and change back dimension to [batch_size, n_step, n_input]
	outputs = tf.pack(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])

	# Hack to build the indexing and retrieve the right output.
	batch_size = tf.shape(outputs)[0]
	# Start indices for each sample
	index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
	# Indexing
	outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

	# Linear activation, using outputs computed above
	return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < training_iters:
		listtt=random.sample(range(0, 24000), batch_size)
		batch_x=[]
		batch_y=[]
		batch_seqlen=[]
		for iiii in range(len(listtt)):
			batch_x.append(trainset[listtt[iiii]])
			batch_y.append(trainset_y[listtt[iiii]])
			batch_seqlen.append(trainset_seqlen[listtt[iiii]])

		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
		                               seqlen: batch_seqlen})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
			                                    seqlen: batch_seqlen})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
			                                 seqlen: batch_seqlen})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
			      "{:.6f}".format(loss) + ", Training Accuracy= " + \
			      "{:.5f}".format(acc))
		step += 1
	print("Optimization Finished!")

	# Calculate accuracy
	test_data = trainset[0:3000]+trainset[15000:18000]
	test_label = trainset_y[0:3000]+trainset_y[15000:18000]
	test_seqlen = trainset_seqlen[0:3000]+trainset_seqlen[15000:18000]
	print("RNN Testing Accuracy:", \
	      sess.run(accuracy, feed_dict={x: test_data, y: test_label,
	                                    seqlen: test_seqlen}))
