import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing


### Uncomment the sections according to the dataset, loss function and fairness constraint (train and test) used, ######

#### The following code will run for crimes dataset, Q_mean loss and DP as the fairness constraint. ######

########## Adult (Protected = 9) ##############################
# df = pd.read_csv('adult_train.csv')
# df_test = pd.read_csv('adult_test.csv')
# data = np.append(df.values, df_test.values, 0)
# #data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')] = 1
# data = np.append(data[:,:-1], l, axis=1)
# data = data[:40000]

######### Compas (Protected = 4) #############################
# df = pd.read_csv('compass.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')] = 1
# data = np.append(data[:,:-1], l, axis=1)
# data = data[:5000]



######## Crimes DATA (Protected -3) #######
df = pd.read_csv('crimes.csv')
data = df.values
l = np.zeros((len(data), 2))
l[np.arange(len(l)), data[:,-1].astype('int')] = 1
data = np.append(data[:,:-1], l, axis=1)[:2000]


########## DEFAULT (Protected = 1) ##############################
# df = pd.read_csv('default.csv')
# data = df.values
# l = np.zeros((len(data), 2))
# l[np.arange(len(l)), data[:,-1].astype('int')] = 1
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



######### Hyperparameters_compass #################
rho = 10
epsilon = 1e-10
num_epochs =1000
batch_size = 1
learning_rate = 0.001
p = 90.0
input_size = (400, data.shape[1]-2) 
hidden_size1 = 500
hidden_size2 = 100
num_classes = 2
temp = 5
prot = -3
############################################
min_max_scaler = preprocessing.MinMaxScaler()
data[:,:input_size[1]] = min_max_scaler.fit_transform(data[:,:input_size[1]])



############## Build Model #####################
input_data = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], input_size[1]), name="data")
labels = tf.placeholder(tf.float32, shape=(batch_size, input_size[0], 2), name = "labels")
fc1_act = dense(input_data, input_size[1], hidden_size1)
fc2_act = dense(fc1_act, hidden_size1, hidden_size2, name = 'dense1')
logits, preds = dense(fc2_act, hidden_size2, num_classes, soft = 1, name = 'dense2')

t_vars = tf.trainable_variables()

# #### LOSS_1 (H_MEAN) ####
# c_00 = tf.reduce_sum(labels[:,:,0])/(epsilon + tf.reduce_sum(tf.multiply(preds[:,:,0] , labels[:,:,0] )))
# c_11 = tf.reduce_sum(labels[:,:,1])/(epsilon + tf.reduce_sum(tf.multiply(preds[:,:,1] , labels[:,:,1] )))
# loss_1 = 1 - 2/ (c_00 + c_11) 

# #### LOSS_2 (Coverage) ####
# c01 = tf.reduce_sum(tf.multiply(preds[:,:,1], labels[:,:,0] ))/(batch_size*input_size[0])
# c11 = tf.reduce_sum(tf.multiply(preds[:,:,1], labels[:,:,1] ))/(batch_size*input_size[0])
# const = c01 + c11  
# loss_2 = tf.maximum(const - 0.25, 0)

# #### LOSS_1 (Q_MEAN) ####
c_00 = (epsilon + tf.reduce_sum(tf.multiply(preds[:,:,0] , labels[:,:,0] )))/tf.reduce_sum(labels[:,:,0])
c_11 = (epsilon + tf.reduce_sum(tf.multiply(preds[:,:,1] , labels[:,:,1] )))/tf.reduce_sum(labels[:,:,1])
loss_1 = tf.sqrt(0.5 * ((1 - c_00)**2 + (1 - c_11)**2))

# #### LOSS_2 (DP) ######
c0 = tf.reduce_sum(tf.multiply(preds[:,:,1], 1- input_data[:,:,4]))/ (batch_size*input_size[0])
c1 = tf.reduce_sum(tf.multiply(preds[:,:,1], input_data[:,:,4]))/(batch_size*input_size[0])
const1 = c0 - c1 
const2 = c1 - c0 
loss_2 = tf.maximum(const1 - 0.20, 0.0) + tf.maximum(const2 - 0.20, 0.0)

#### LOSS_1 (F_MEASURE) ####
# c_00 = tf.reduce_sum(tf.multiply(labels[:,:,0], preds[:,:,0] ))
# c_11 = tf.reduce_sum(tf.multiply(labels[:,:,1], preds[:,:,1] ))
# c_01 = tf.reduce_sum(tf.multiply(labels[:,:,0], preds[:,:,1] ))
# c_10 = tf.reduce_sum(tf.multiply(labels[:,:,1], preds[:,:,0] ))
# loss_1 = 1 - (2*c_11/ (2*c_11 + c_01 + c_10))

#### LOSS_2 (KLD) ######
# pi_0 = tf.reduce_sum(labels[:,:,0])
# pi_1 = tf.reduce_sum(labels[:,:,1]) 
# const = ( pi_0 * tf.log(pi_0/(c_00 + c_10)) + pi_1 * tf.log(pi_1/(c_01 + c_11)) )/(batch_size*input_size[0])
# loss_2 = tf.maximum(const - 0.001, 0.0)


#### Overall loss and optimizer ####
loss  = loss_1 + (rho * loss_2**2)  
#loss = loss_1
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = t_vars)



optim = tf.train.AdamOptimizer(learning_rate)
gvs = optim.compute_gradients(loss_1, t_vars)

##################################################


mean_acc = []

for cv in range(5):

	t_ix = np.random.randint(0, len(data), int(0.7*len(data)))
	train_data = data[t_ix]
	test_data = data[np.random.randint(0, len(data), int(0.3*len(data)))]

	################ TRAIN ################################
	tf.global_variables_initializer().run(session=sess)
	saver = tf.train.Saver(max_to_keep=1)
	#saver.restore(sess, '/Neutron6/manisha.padala/game_theroy/-2')
	train_loss=[]
	Loss1= []
	Loss2 = []
	start_time = time.time()
	for epoch in range(1,num_epochs):
		h_loss = []
		cov_loss = []
		batches = len(train_data)/input_size[0]
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
			h_loss.append(l1)
			cov_loss.append(l2)
		if epoch%1 == 0 or epoch == 1:
			train_loss.append(l)
			Loss1.append(np.mean(h_loss))
			#Loss2.append(np.mean(cov_loss))
			# plt.plot(train_loss)
			# plt.savefig('results/Overall_loss.jpg')
			# plt.close()
			# plt.plot(Loss1)
			# plt.savefig('results/Log_loss.jpg')
			# plt.close()
			# plt.plot(Loss2)
			# plt.savefig('results/DI_loss.jpg')
			# plt.close()
			#print(np.mean(h_loss), np.mean(cov_loss))
	
	################ TEST #################################
	#print(di_/batches)

	labels_predicted =[]
	total = 0
	cov = 0
	batches = len(test_data)/input_size[0]
	for itr in range(batches):
			data_ = test_data[itr*input_size[0]:(itr+1)*input_size[0], :input_size[1]].reshape(batch_size, input_size[0], input_size[1])
			true_label = test_data[itr*input_size[0]:(itr+1)*input_size[0], input_size[1]:].reshape(batch_size, input_size[0], num_classes)
			h_l, pred, con1, con2 = sess.run([loss_1, preds, const, const],feed_dict={ input_data: data_, labels: true_label})
			labels_predicted.extend(np.argmax(pred[0],1).flatten())
			cov += np.maximum(con1, con2)
			total += h_l
	print('Constraint', cov/batches)
	labels_predicted = np.array(labels_predicted)
	
	# ### H_MEAN ###
	# ix = np.where(test_data[:,-1]==1)[0]
	# ix_0 = np.where(test_data[:,-2] == 1)[0]
	# term1 = len(ix)/(np.sum(labels_predicted[ix.astype('int')]).astype('float') + epsilon)
	# term0 = len(ix_0)/(np.sum(1-labels_predicted[ix_0.astype('int')]).astype('float') + epsilon )
	# hloss_test = 1 - 2/(term1+term0)
	# ### COV ###
	# cov_test = (np.sum(test_data[:,-2]*labels_predicted) + np.sum(test_data[:,-1]*labels_predicted) ) / len(test_data)
	
	# ## Q_MEAN ###
	ix = np.where(test_data[:,-1]==1)[0]
	ix_0 = np.where(test_data[:,-2] == 1)[0]
	term1 = np.sum(labels_predicted[ix.astype('int')]).astype('float')/len(ix)
	term0 = np.sum(1-labels_predicted[ix_0.astype('int')]).astype('float')/len(ix_0)
	qloss_test = np.sqrt( 0.5 * ((1 - term0)**2 + (1 - term1)**2)) 

	# ### DP ###
	t1 = np.sum(labels_predicted * test_data[:,4])/ len(test_data) 
	t0 = np.sum(labels_predicted * (1 - test_data[:,4]))/ len(test_data)
	dp_test = np.maximum(t0 -t1, t1-t0)

	### F_MEASURE ###
	# t_00 = np.sum(test_data[:,-2]* (1 - labels_predicted))
	# t_11 = np.sum(test_data[:,-1]* labels_predicted)
	# t_01 = np.sum(test_data[:,-2]* labels_predicted)
	# t_10 = np.sum(test_data[:,-1]* (1 - labels_predicted))
	# f_measure_test = 1 - (2*t_11/(2*t_11 + t_10 + t_01))

	### KLD ###
	# p_0 = np.sum(test_data[:, -2])
	# p_1 = np.sum(test_data[:, -1])
	# kld_test= (p_0 * np.log(p_0/(t_00 + t_10)) + p_1 * np.log(p_1/(t_01 + t_11))) / len(test_data)

	print('Loss of the network, Constraint', qloss_test, dp_test)
	




