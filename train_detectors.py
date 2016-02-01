import cPickle
import tensorflow as tf
import sklearn
import sklearn.svm
import gabor
import matplotlib.pyplot as plt
import numpy as np
import pandas

NUM_EPOCHS = 1000

def conv2d (x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool (x, size=2):
	return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def filterFaces (faces, filterBankF):
	CHUNK = 1000
	results = []
	for i in range(int(np.ceil(faces.shape[0]/float(CHUNK)))):
		idxs = range(i*CHUNK, min((i+1)*CHUNK, faces.shape[0]))
		someResults = filterSomeFaces(faces[idxs, :, :], filterBankF)
		results.append(someResults)
	return np.vstack(results)

def filterSomeFaces (faces, filterBankF):
	facesF = np.tile(np.fft.fft2(faces).reshape((faces.shape[0], 1, faces.shape[1], faces.shape[2])), (1, len(filterBankF), 1, 1))
	kernelsF = np.tile(filterBankF.reshape((1,) + filterBankF.shape), (faces.shape[0], 1, 1, 1))
	resultsF = facesF * np.conjugate(kernelsF)
	del facesF
	del kernelsF
	results = np.abs(np.fft.ifft2(resultsF))
	del resultsF
	norm = np.linalg.norm(results, axis=(2,3)).reshape((results.shape[0], results.shape[1], 1, 1))
	norm = np.tile(norm, (1, 1, faces.shape[1], faces.shape[2]))
	results /= norm
	del norm
	results = np.reshape(results.astype(np.float32), (faces.shape[0], len(filterBankF)*faces.shape[1]*faces.shape[2]))
	return results

def getData ():
	(allFrameFilenames, allFaces) = cPickle.load(open("faces.pkl", "rb"))
	allFrameFilenames = np.array(allFrameFilenames)
	faces = []
	labels = []
	isVSU = []
	subjects = []
	d = pandas.io.parsers.read_csv('labels.txt', sep='\t')
	e = d.groupby("frame_filename")
	for frameFilename in e.indices.keys():
		idxs = e.indices[frameFilename]
		engagementLabels = d.engagement.iloc[idxs]
		minLabel = np.min(engagementLabels)
		subjectId = d.subject_id.iloc[idxs[0]]
		if minLabel == -1:
			continue
		maxLabel = np.max(engagementLabels)
		if maxLabel - minLabel > 1:
			continue
		avgLabel = int(round(np.mean(engagementLabels)))
		idx = np.nonzero(allFrameFilenames == frameFilename)[0]
		if len(idx) > 0:
			faces.append(allFaces[idx,:,:].squeeze())
			labels.append(avgLabel)
			isVSU.append("UCSD" not in frameFilename)
			subjects.append(subjectId)
	faces = np.array(faces)
	labels = np.array(labels)
	isVSU = np.array(isVSU)
	subjects = np.array(subjects)
	return faces, labels, isVSU, subjects

def trainNN (faces, labels, subjects, e):
	#faces -= np.tile(np.reshape(np.mean(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,48,48])
	#faces /= np.tile(np.reshape(np.std(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,48,48])

	uniqueSubjects = np.unique(subjects)
	accuracies = []
	for testSubject in uniqueSubjects:
		idxs = np.nonzero(subjects != testSubject)[0]
		train_x = faces[idxs]
		train_x = np.reshape(train_x, train_x.shape + (1,))
		someLabels = labels[idxs]
		train_y = np.atleast_2d(someLabels != e).T
		train_y = np.hstack((train_y, 1 - train_y)).astype(np.float32)
		mx = np.mean(train_x, axis=0, keepdims=True)
		sx = np.std(train_x, axis=0, keepdims=True)
		sx[sx == 0] = 1
		# Normalize
		#train_x = (train_x - mx) / sx

		# Shuffle training examples
		idxs = np.random.permutation(train_x.shape[0])
		train_x = train_x[idxs,:,:,:]
		train_y = train_y[idxs,:]

		idxs = np.nonzero(subjects == testSubject)[0]
		test_x = faces[idxs]
		test_x = np.reshape(test_x, test_x.shape + (1,))
		someLabels = labels[idxs]
		test_y = np.atleast_2d(someLabels != e).T
		test_y = np.hstack((test_y, 1 - test_y)).astype(np.float32)
		# Normalize
		test_x = (test_x - mx) / sx

		auc = runNNSimple(train_x, train_y, test_x, test_y)
		print "{}: {}".format(testSubject, auc)
		print ""

def trainSVM (filteredFaces, labels, subjects, e):
	uniqueSubjects = np.unique(subjects)
	accuracies = []
	masterK = filteredFaces.dot(filteredFaces.T)
	for testSubject in uniqueSubjects:
		idxs = np.nonzero(subjects != testSubject)[0]
		someFilteredFacesTrain = filteredFaces[idxs]
		someLabels = labels[idxs]
		y = someLabels == e
		K = masterK[idxs, :]
		K = K[:, idxs]
		svm = sklearn.svm.SVC(kernel="precomputed")
		svm.fit(K, y)

		idxs = np.nonzero(subjects == testSubject)[0]
		someFilteredFaces = filteredFaces[idxs]
		someLabels = labels[idxs]
		y = someLabels == e
		yhat = svm.decision_function(someFilteredFaces.dot(someFilteredFacesTrain.T))

		if len(np.unique(y)) > 1:
			auc = sklearn.metrics.roc_auc_score(y, yhat)
		else:
			auc = np.nan
		print "{}: {}".format(testSubject, auc)
		accuracies.append(auc)
	accuracies = np.array(accuracies)
	accuracies = accuracies[np.isfinite(accuracies)]
	print np.mean(accuracies), np.median(accuracies)

def weight_variable (shape, wd = 0):
	var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
	tf.add_to_collection("losses", weight_decay)
	return var

def bias_variable (shape):
	var = tf.constant(0.1, shape=shape)
	return tf.Variable(var)

def runNNSimple (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	BATCH_SIZE = 128
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		W_conv1 = weight_variable([7, 7, 1, 4])
		b_conv1 = bias_variable([4])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool(h_conv1, 4)
		h_pool1_reshaped = tf.reshape(h_pool1, [-1, 12*12*4])

		# FC2
		W2 = weight_variable([ 12*12*4, train_y.shape[1] ], wd=1e-1)
		b2 = bias_variable([ train_y.shape[1] ])
		y_conv = tf.nn.softmax(tf.matmul(h_pool1_reshaped, W2) + b2)

		cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		#train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(total_loss)
		LEARNING_RATE = 0.05
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/5, 0.95, staircase=True)
		train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(total_loss, global_step=batch)
		#train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			train_step.run({x_image: train_x[offset:offset+BATCH_SIZE, :, :, :], y_: train_y[offset:offset+BATCH_SIZE, :]})
			if i % 50 == 0:
				ll = cross_entropy.eval({x_image: train_x, y_: train_y})
				auc = sklearn.metrics.roc_auc_score(train_y[:,1], y_conv.eval({x_image: train_x})[:,1])
				print "Train LL={} AUC={}".format(ll, auc)

				ll = cross_entropy.eval({x_image: test_x, y_: test_y})
				auc = sklearn.metrics.roc_auc_score(test_y[:,1], y_conv.eval({x_image: test_x})[:,1])
				print "Test LL={} AUC={}".format(ll, auc)

				plt.imshow(np.reshape(h_conv1.eval({x_image: train_x[0:1,:,:,:]})[0,:,:,0], [48,48]))
				plt.show()
				plt.imshow(np.reshape(h_pool1.eval({x_image: train_x[0:1,:,:,:]})[0,:,:,0], [12,12]))
				plt.show()
				plt.imshow(np.reshape(W2.eval()[0:144,0], [ 12,12 ]))
				plt.show()
		auc = sklearn.metrics.roc_auc_score(test_y[:,1], y_conv.eval({x_image: test_x})[:,1])
		session.close()
		return auc

def runNN (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	BATCH_SIZE = 128
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		W_conv1 = weight_variable([7, 7, 1, 4])
		b_conv1 = bias_variable([4])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool(h_conv1, 8)

		W_fc1 = weight_variable([6 * 6 * 4, 4])
		b_fc1 = bias_variable([4])
		h_pool1_flat = tf.reshape(h_pool1, [-1, 6*6*4])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		W_fc2 = weight_variable([4, train_y.shape[1]])
		b_fc2 = bias_variable([train_y.shape[1]])

		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		LEARNING_RATE = 0.1
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/5, 0.95, staircase=True)
		train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1).minimize(total_loss, global_step=batch)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			train_step.run({x_image: train_x[offset:offset+BATCH_SIZE, :, :, :], y_: train_y[offset:offset+BATCH_SIZE, :], keep_prob: 0.5})
			if i % 10 == 0:
				print sklearn.metrics.roc_auc_score(train_y[:,1], y_conv.eval({x_image: train_x, keep_prob: 1.0})[:,1])
			if i % 50 == 0:
				ll = cross_entropy.eval({x_image: test_x, y_: test_y, keep_prob: 1.0})
				auc = sklearn.metrics.roc_auc_score(test_y[:,1], y_conv.eval({x_image: test_x, keep_prob: 1.0})[:,1])
				print "Test LL={} AUC={}".format(ll, auc)
		session.close()

if __name__ == "__main__":
	if 'faces' not in globals():
		faces, labels, isVSU, subjects = getData()
	
		# Restrict to VSU
		faces = faces[np.nonzero(isVSU)]
		labels = labels[np.nonzero(isVSU)]
		subjects = subjects[np.nonzero(isVSU)]

		filterBank = gabor.makeGaborFilterBank(faces.shape[-1])
		filterBankF = np.fft.fft2(filterBank)
		filteredFaces = filterFaces(faces, filterBankF)

	for e in [ 1, 2, 3, 4 ]:  # Engagement label
		print "E={}".format(e)
		#trainSVM(filteredFaces, labels, subjects, e)
		trainNN(faces, labels, subjects, e)
