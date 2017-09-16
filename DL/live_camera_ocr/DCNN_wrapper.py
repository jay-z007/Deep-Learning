import tensorflow as tf
import numpy as np

"""
TODO : consider not to train the locally connected layers if the digit is absent (hint : use length)
TODO : 
"""

class DCNN_model(object):
	"""docstring for DCNN_model"""
	def __init__(self, valid_dataset, test_dataset, test_labels, image_size=32, num_channels=1, num_labels=11, 
		lr=0.001, batch_size=16, patch_size=3, depth=[16,32,64], num_hidden=512, ckpt_path="./model_v2/", 
		model_name="DCNN_model"):#, test_dataset):
		
		super(DCNN_model, self).__init__()

		self.image_size = image_size
		self.num_channels = num_channels
		self.num_labels = num_labels		
		self.lr = lr
		self.batch_size = batch_size
		self.patch_size = patch_size
		self.depth = depth
		self.num_hidden = num_hidden
		self.ckpt_path = ckpt_path
		self.model_name = model_name
		self.test_labels = test_labels

		def __graph__():

			graph = tf.Graph()

			with graph.as_default():

				# Input data.

				self.tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, image_size, image_size, num_channels))
				self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 6, num_labels))
				self.tf_valid_dataset = tf.constant(valid_dataset)
				self.tf_test_dataset = tf.constant(test_dataset)
				beta = 0.001
				
				weights = {
				'conv_1' : tf.Variable(tf.truncated_normal(
					[patch_size, patch_size, num_channels, depth[0]], stddev=0.1)),

				'conv_2' : tf.Variable(tf.truncated_normal(
					[patch_size, patch_size, depth[0], depth[1]], stddev=0.1)),

				'conv_3' : tf.Variable(tf.truncated_normal(
					[patch_size, patch_size, depth[1], depth[2]], stddev=0.1)),

				# 'conv_4' : tf.Variable(tf.truncated_normal(
				# 	[patch_size, patch_size, depth[2], depth[3]], stddev=0.1)),

				'fc_1' : tf.Variable(tf.truncated_normal(
					[(image_size // (2**len(depth))) * (image_size // (2**len(depth))) * depth[2], num_hidden], 
					stddev=0.1)),

				'L' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1)),

				's1' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1)),

				's2' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1)),

				's3' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1)),

				's4' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1)),				

				's5' : tf.Variable(tf.truncated_normal(
					[num_hidden, num_labels], stddev=0.1))

				}

				digit_weights = [
						weights["s1"],
						weights["s2"],
						weights["s3"],
						weights["s4"],
						weights["s5"],
					]

				biases = {
				'conv_1' : tf.Variable(tf.zeros([depth[0]])),

				'conv_2' : tf.Variable(tf.constant(1.0, shape=[depth[1]])),

				'conv_3' : tf.Variable(tf.constant(1.0, shape=[depth[2]])),

				# 'conv_4' : tf.Variable(tf.constant(1.0, shape=[depth[3]])),

				'fc_1' : tf.Variable(tf.constant(1.0, shape=[num_hidden])),

				'L' : tf.Variable(tf.constant(1.0, shape=[num_labels])),

				's1' : tf.Variable(tf.constant(1.0, shape=[num_labels])),

				's2' : tf.Variable(tf.constant(1.0, shape=[num_labels])),

				's3' : tf.Variable(tf.constant(1.0, shape=[num_labels])),

				's4' : tf.Variable(tf.constant(1.0, shape=[num_labels])),				

				's5' : tf.Variable(tf.constant(1.0, shape=[num_labels]))

				}

				self.keep_prob = [tf.placeholder(tf.float32) for _ in range(4)]
				
				# compute.
				def compute(data):
					conv = tf.nn.conv2d(data, weights["conv_1"], [1, 1, 1, 1], padding='SAME')
					hidden = tf.nn.relu(conv + biases["conv_1"])
					pool = tf.nn.dropout(tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'),
						keep_prob=self.keep_prob[0])


					conv = tf.nn.conv2d(pool, weights["conv_2"], [1, 1, 1, 1], padding='SAME')
					hidden = tf.nn.relu(conv + biases["conv_2"])
					pool = tf.nn.dropout(tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'),
						keep_prob=self.keep_prob[1])
					
					conv = tf.nn.conv2d(pool, weights["conv_3"], [1, 1, 1, 1], padding='SAME')
					hidden = tf.nn.relu(conv + biases["conv_3"])
					pool = tf.nn.dropout(tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'),
						keep_prob=self.keep_prob[2])

					# conv = tf.nn.conv2d(pool, weights["conv_4"], [1, 1, 1, 1], padding='SAME')
					# hidden = tf.nn.relu(conv + biases["conv_4"])
					# pool = tf.nn.dropout(tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'),
					# 	keep_prob=self.keep_prob)

					shape = pool.get_shape().as_list()
					reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
					hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weights["fc_1"]) + biases["fc_1"]),
						keep_prob=self.keep_prob[3])

					#logits_L = tf.matmul(hidden, weights["L"]) + biases["L"]
					logits_s1 = tf.matmul(hidden, weights["s1"]) + biases["s1"]
					logits_s2 = tf.matmul(hidden, weights["s2"]) + biases["s2"]
					logits_s3 = tf.matmul(hidden, weights["s3"]) + biases["s3"]
					logits_s4 = tf.matmul(hidden, weights["s4"]) + biases["s4"]
					logits_s5 = tf.matmul(hidden, weights["s5"]) + biases["s5"]

					return [logits_s1, logits_s2, logits_s3, logits_s4, logits_s5]

					#return [logits_L, logits_s1, logits_s2, logits_s3, logits_s4, logits_s5]
				
				# Training computation.
				logits = compute(self.tf_train_dataset)

				self.loss = tf.add_n([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
					labels=self.tf_train_labels[:, i+1], logits=logits[i])) + beta*tf.nn.l2_loss(digit_weights[i])
				for i in range(5)])

				global_step = tf.Variable(0)  # count the number of steps taken.
				start_learning_rate = self.lr
				learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 2500, 0.8, staircase=True)
				self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

				# Predictions for the training, validation, and test data.
				def prediction(dataset):
					return tf.stack([
					tf.nn.softmax(compute(dataset)[0]), 
					tf.nn.softmax(compute(dataset)[1]), 
					tf.nn.softmax(compute(dataset)[2]), 
					tf.nn.softmax(compute(dataset)[3]), 
					tf.nn.softmax(compute(dataset)[4])])#, 
					#tf.nn.softmax(compute(dataset)[5])])

				self.train_prediction = tf.stack([
					tf.nn.softmax(logits[0]), 
					tf.nn.softmax(logits[1]), 
					tf.nn.softmax(logits[2]), 
					tf.nn.softmax(logits[3]), 
					tf.nn.softmax(logits[4])])#, 
					#tf.nn.softmax(logits[5])])

				self.valid_prediction = tf.nn.softmax(compute(self.tf_valid_dataset))
				self.test_prediction = tf.nn.softmax(compute(self.tf_test_dataset))

			return graph

		print "Building Graph"
		self.graph = __graph__()


	def train(self, train_dataset, train_labels, valid_dataset, valid_labels):

		num_steps = 15001

		with tf.Session(graph=self.graph) as sess:
			tf.global_variables_initializer().run()

			# we need to save the model periodically
			saver = tf.train.Saver()
			
			ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
			# restore session
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

			print('Initialized')
			for step in range(num_steps):
				offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
				batch_data = train_dataset[offset:(offset + self.batch_size), :, :, :]
				batch_labels = train_labels[offset:(offset + self.batch_size), :]
				
				feed_trn_kp = [0.9, 0.75, 0.75, 0.5]
				feed_trn_kp = {i:v for i,v in zip(self.keep_prob, feed_trn_kp)}


				feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
				feed_dict.update(feed_trn_kp)

				_, l, predictions = sess.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
		
				if (step % 500 == 0):
					saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=step)
					print('Minibatch self.loss at step %d: %f' % (step, l))
					#print predictions
					print('Minibatch single digit accuracy: %.1f%%' % self.accuracy_single(predictions, batch_labels))
					print('Minibatch accuracy: %.1f%%' % self.accuracy(predictions, batch_labels))
					#predictions = self.predict(sess, valid_dataset)
					feed_dict = {i : 1.0 for i in self.keep_prob}
					# print('Validation accuracy: %.1f%%' % self.accuracy(predictions, valid_labels))
					print('Validation single digit accuracy: %.1f%%' % self.accuracy_single(self.valid_prediction.eval(feed_dict=feed_dict), valid_labels))
					print('Validation accuracy: %.1f%%' % self.accuracy(self.valid_prediction.eval(feed_dict=feed_dict), valid_labels))
					
			print('Test single digit accuracy: %.1f%%' % self.accuracy_single(self.test_prediction.eval(feed_dict=feed_dict), self.test_labels))
			print('Test accuracy: %.1f%%' % self.accuracy(self.test_prediction.eval(feed_dict=feed_dict), self.test_labels))

	"""
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%	
		%	TODO : change this function to accomodate the multidigits
		%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	"""
	# def single_accuracy(self, predictions, labels):
	# 	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	def accuracy_single(self, predictions, labels):	
		a = np.argmax(predictions, 2).T == np.argmax(labels[:,1:6], 2)
		length = np.argmax(labels[:,0], axis=1)
		summ = 0.0
		for i in range(len(length)):
			summ += np.sum(a[i,:length[i]])
		return(100 * summ / np.sum(length))


	def accuracy(self, predictions, labels):
		total = predictions.shape[1]
		# print total 
		# print np.argmax(labels, 2)
		# print np.argmax(predictions, 2).T
		# print np.argmax(predictions, 2).T == np.argmax(labels, 2)
		# print np.all(np.argmax(predictions, 2).T == np.argmax(labels, 2), axis=1)
		# acc = 100.0 * np.sum(np.all(np.argmax(predictions, 2).T == np.argmax(labels, 2), axis=1))/total
		# return  acc
		count = predictions.shape[1]
		return 100.0 * (count - np.sum([1 for i in np.argmax(predictions, 2).T == np.argmax(labels[:,1:6], 2) if False in i])) / count

	"""
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%	
		%	TODO : 
		%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	"""
	def predict(self, sess, dataset):
		
		predictions = []		
		for i in range(0, len(dataset)/self.batch_size, self.batch_size):
			batch_data = dataset[i : i+self.batch_size]
			feed_dict = {self.tf_train_dataset : batch_data}#, self.tf_train_labels : batch_labels}
			p = sess.run(self.train_prediction, feed_dict=feed_dict)
			predictions.extend(p)

		return np.array(predictions)
