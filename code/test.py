import tensorflow as tf ### Version (pip install tensorflow-gpu==1.15) (python2)
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
from numpy import linalg as LA

### Uncomment the sections according to the dataset, fairness constraint (train and test) used, ######

#### The following code will run for compass dataset with Equalized odds as the fairness constraint. ######

######### Adult (Protected = 9) ##############################
#df = pd.read_csv('adult_train.csv')
#df_test = pd.read_csv('adult_test.csv')
#data = np.append(df.values, df_test.values, 0)
#l = np.zeros((len(data), 2))
#l[np.arange(len(l)), data[:,-1].astype('int')] = 1
#data = np.append(data[:,:-1], l, axis=1)
#data = data[:40000]


########### BANK (Protected = 0) ##############################
# df = pd.read_csv('bank_train_new.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')] = 1
# data = np.append(data[:,:-1], l, axis=1)
# data = data[:40000]

######### Compas (Protected = 4) #############################
df = pd.read_csv('compass.csv')
data = df.values
l = np.zeros((len(data), 2))
l[np.arange(len(l)), data[:,-1].astype('int')] = 1
data = np.append(data[:,:-1], l, axis=1)
data = data[:5000]


######### German (Protected = 8) #############################
# df = pd.read_csv('german.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')-1] = 1
# data = np.append(data[:,:-1], l, axis=1)



#############################################
print(data.shape)


################### Neurel Network Layers  ####################

#### Layer 1 ###
def dense(inp, inp_shape, hidden_size, soft = 0, name ='dense'):
	with tf.variable_scope(name):
		weights = tf.get_variable("weights", [1, inp_shape, hidden_size], 'float32',initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
		bias = tf.get_variable("bias", [hidden_size], 'float32', initializer=tf.constant_initializer(0, dtype=tf.float32))
		weights = tf.tile(weights, (batch_size, 1, 1))
		out = tf.matmul(inp, weights) + bias
		if soft == 0:
			return tf.nn.relu(out)
		else:
			return out, tf.nn.softmax(temp*out)
#################




sess = tf.Session()

######### Hyperparameters #################
rho = 10
epsilon = 1e-10
num_epochs =1000
batch_size = 1
learning_rate = 0.001
p = 90.0 ## change for different values of p (for DI)
input_size = (500, data.shape[1]-2) 
hidden_size1 = 500
hidden_size2 = 100
num_classes = 2
temp = 5
prot = 4  ## Change the prot based on the protected of the corresponding datatset
############################################
min_max_scaler = preprocessing.MinMaxScaler()
data[:,:input_size[1]] = min_max_scaler.fit_transform(data[:,:input_size[1]])




############## Build Model #####################
input_data = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], input_size[1]), name="data")
labels = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], 2), name = "labels")
#prox_inp = tf.concat([input_data[:,:,:9], input_data[:,:,10:]], axis = 2)
fc1_act = dense(input_data, input_size[1], hidden_size1)
fc2_act = dense(fc1_act, hidden_size1, hidden_size2, name = 'dense1')
logits, rounded = dense(fc2_act, hidden_size2, num_classes, soft = 1, name = 'dense2')
lag = tf.get_variable("lag", (), 'float32', initializer=tf.constant_initializer(rho, dtype=tf.float32))

t_vars = tf.trainable_variables()[:-1]

#loss_1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=4.1))
loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

############ (Uncomment the followinf for traning with) Disparate Impact ####################################
# n_r = tf.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]))
# d_r = tf.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot]))
# n_d = n_r / (d_r + epsilon)
# n_d_ = d_r / (n_r + epsilon)
# const = tf.reduce_min(tf.minimum(n_d, n_d_)) - p/100.0
# loss_2 = tf.maximum(-const,0)
###################################


############ (Uncomment the following for traning with) Demographic parity ####################################
#c0 = tf.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]))/ (batch_size*input_size[0])
#c1 = tf.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot]))/(batch_size*input_size[0])
#const1 = c0 - c1 
#loss_2 = tf.maximum(const1 - 0.010, 0.0) + tf.maximum(-const1 - 0.010, 0.0)
#loss_2 = tf.abs(const1) - 0.05

############ (Uncomment the following for traning with) Demographic Parity Modified ####################################
# c = tf.reduce_sum(rounded[:,:,1]) /(batch_size*input_size[0])
# c0 = tf.reduce_sum(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]))/ (tf.reduce_sum(1- input_data[:,:,prot]))
# c1 = tf.reduce_sum(tf.multiply(rounded[:,:,1], input_data[:,:,prot]))/(tf.reduce_sum(input_data[:,:,prot]))


########### (Uncomment the following for traning with) Equalized Odds ####################################
c10_0 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,1], 1- input_data[:,:,prot]), labels[:,:,0]))/ tf.reduce_sum(1- input_data[:,:,prot])
c10_1 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,1], input_data[:,:,prot]), labels[:,:,0]))/tf.reduce_sum(input_data[:,:,prot])
c01_0 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,0], 1- input_data[:,:,prot]), labels[:,:,1]))/ tf.reduce_sum(1- input_data[:,:,prot])
c01_1 = tf.reduce_sum(tf.multiply(tf.multiply(rounded[:,:,0], input_data[:,:,prot]), labels[:,:,1]))/tf.reduce_sum(input_data[:,:,prot])
const1 = tf.abs(c10_0 - c10_1) 
const2 = tf.abs(c01_0 - c01_1)
loss_2 = tf.maximum(tf.maximum(const1, const2) - 0.01, 0.0)
# loss_2 = tf.maximum(const1, const2) - 0.04
####################################################################


loss  = loss_1 + (lag * loss_2)  
#loss = loss_1
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = t_vars)
opt1 = tf.train.AdamOptimizer(learning_rate).minimize(-loss, var_list = [lag])

# optim = tf.train.AdamOptimizer(learning_rate)
# gvs = optim.compute_gradients(loss, t_vars)
# grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape(), name="grad_placeholder"), grad[1]) for grad in gvs[:-1]]
# train_step = optim.apply_gradients(grad_placeholder)

##################################################

mean_acc = []
dp_mean = 0
for cv in range(5):

	t_ix = np.random.randint(0, len(data), int(0.8*len(data)))
	train_data = data[t_ix]
	test_data = data[np.random.randint(0, len(data), int(0.2*len(data)))]

	################ TRAIN ################################
	tf.global_variables_initializer().run(session=sess)
	saver = tf.train.Saver(max_to_keep=1)
	#saver.restore(sess, '/Neutron6/manisha.padala/game_theroy/-2')
	start_time = time.time()
	for epoch in range(1,num_epochs):
		log_loss = []
		di_loss = []
		batches = int(len(train_data)/input_size[0])
		for itr in range(batches):
			data_ = train_data[itr*input_size[0]:(itr+1)*input_size[0], :input_size[1]].reshape(batch_size, input_size[0], input_size[1])
			true_label = train_data[itr*input_size[0]:(itr+1)*input_size[0], input_size[1]:].reshape(batch_size, input_size[0], num_classes)
			dict1 = { input_data: data_, labels: true_label}
			# g = sess.run(gvs[:-1],feed_dict= dict1)
			# dict2 = { i[0]: d[0] for i, d in zip(grad_placeholder, g)}
			# dict1.update(dict2)
			# _,_,_,l1, l2, l = sess.run([train_step, op1, op2, loss_1, loss_2, loss],feed_dict= dict1)
			#print(w)
			_, l1, l2, l = sess.run([opt, loss_1, loss_2, loss], feed_dict = dict1)
			_, lam, l2, l = sess.run([opt1, lag, loss_2, loss], feed_dict = dict1)
			
		#if epoch%1000 == 0 or epoch == 1:
		#	print(lam)
			
			
	
	################ TEST #################################
	#print(di_/batches)
	plot_data = np.zeros((len(test_data), 3))
	labels_predicted  = []
	correct = 0
	total = 0
	di_ = 0
	batches = int(len(test_data)/input_size[0])
	for itr in range(batches):
			data_ = test_data[itr*input_size[0]:(itr+1)*input_size[0], :input_size[1]].reshape(batch_size, input_size[0], input_size[1])
			true_label = test_data[itr*input_size[0]:(itr+1)*input_size[0], input_size[1]:].reshape(batch_size, input_size[0], num_classes)
			pred_ = sess.run([rounded],feed_dict={ input_data: data_, labels: true_label})
			#print(di, 1.0/(di+epsilon))
			#di_ += 1.0/(di+epsilon)
			pred_ = pred_[0]
			labels_predicted.extend(np.argmax(pred_[0],1).flatten())
			plot_data[itr*input_size[0]:(itr+1)*input_size[0], 0] = data_[0,:,4]
			plot_data[itr*input_size[0]:(itr+1)*input_size[0], 1] = np.argmax(true_label[0],1).flatten()
			plot_data[itr*input_size[0]:(itr+1)*input_size[0], 2] = np.argmax(pred_[0],1).flatten()
			correct += list(np.argmax(pred_[0],1).flatten() == np.argmax(true_label[0],1).flatten()).count(True)
			total += len(true_label[0])
                        


	labels_predicted = np.array(labels_predicted)
	# DP ###
	#t1 = np.sum(labels_predicted * test_data[:,prot])/ len(test_data) 
	#t0 = np.sum(labels_predicted * (1 - test_data[:,prot]))/ len(test_data)
	#dp_test = np.maximum(t0 -t1, t1-t0)
	#dp_mean += dp_test
	np.save('plot_'+ str(cv)+'.npy', plot_data)

	#### DP Modified ########
	# t = np.sum(labels_predicted)/ np.float(len(test_data))
	# t1 = np.sum(labels_predicted * test_data[:,prot])/ np.sum(test_data[:, prot]) 
	# t0 = np.sum(labels_predicted * (1 - test_data[:,prot]))/ np.sum(1 - test_data[:,prot])
	# dp_test = np.maximum(np.abs(t -t1), np.abs(t-t0))
	# dp_mean += dp_test

	## EO ###
	t10_1 = np.sum(test_data[:,-2] * labels_predicted * test_data[:,prot])/ np.sum((test_data[:,prot])) 
	t10_0 = np.sum(test_data[:,-2] * labels_predicted * (1 - test_data[:,prot]))/ np.sum((1 - test_data[:,prot]))
	t01_1 = np.sum(test_data[:,-1] * (1 - labels_predicted) * test_data[:,prot])/ np.sum((test_data[:,prot])) 
	t01_0 = np.sum(test_data[:,-1] * (1 - labels_predicted) * (1 - test_data[:,prot]))/ np.sum((1 - test_data[:,prot]))
	eo_test = np.maximum(np.abs(t10_1 - t10_0), np.abs(t01_1 - t01_0))
	dp_mean += eo_test

	#### DI ###########
	# nr = np.sum(labels_predicted * test_data[:,prot])
	# dr = np.sum(labels_predicted * (1 - test_data[:,prot]))
	# print('DI value of the prediction:', nr/dr, dr/nr)
	#print('DP of the test', eo_test)
	#dp_mean += np.minimum(nr/dr, dr/nr)

	print(eo_test)
	print('Accuracy of the network: {} %'.format(100.0 * correct / total))
	mean_acc.append((100.0 * correct / total))


print('Mean accuracy of the network, dp',np.mean(mean_acc), dp_mean/5.0)





