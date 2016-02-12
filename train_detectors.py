import cPickle
import sys
import tensorflow as tf
import sklearn
import sklearn.svm
import gabor
import matplotlib.pyplot as plt
import numpy as np
import pandas
from skimage.transform import resize

FACE_SIZE = 48
NUM_EPOCHS = 20000
BATCH_SIZE = 128

def resizeFaces (faces, newSize):
	newFaces = np.zeros((faces.shape[0], newSize, newSize), dtype=np.float32)
	for i in range(faces.shape[0]):
		newFaces[i,:,:] = resize(faces[i,:,:], (newSize, newSize), preserve_range=True)
	return newFaces

def getEigenfaces (faces):
	faces = np.reshape(faces, [ faces.shape[0], faces.shape[1]*faces.shape[2]*faces.shape[3] ]).copy()  # Copy so we don't affect caller's object
	faces -= np.mean(faces, axis=0, keepdims=True)
	cov = faces.T.copy().dot(faces)  # Copy so that it's contiguous and thus faster
	u,s,v = np.linalg.svd(cov)
	return u[:, 0:NUM_COMPONENTS]

def perturbFaces (faces, numFaces):
	eigenfaces = getEigenfaces(faces)
	coefficients = faces.reshape([ faces.shape[0], faces.shape[1]*faces.shape[2]*faces.shape[3] ]).dot(eigenfaces)
	stds = np.std(coefficients, axis=0, keepdims=True)
	perturbations = eigenfaces.dot(np.random.randn(eigenfaces.shape[1], numFaces) * stds.T).T.reshape([ numFaces, faces.shape[1], faces.shape[2], 1 ])
	return faces[0:numFaces,:,:,:] + perturbations

def conv2d (x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool (x, size=2):
	return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def filterFaces (faces, filterBankF):
	CHUNK = 1000
	results = []
	for i in range(int(np.ceil(faces.shape[0]/float(CHUNK)))):
		print "Chunk {}".format(i)
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

def getDataFast ():
	if FACE_SIZE == 36:
		faces = np.load("faces36.npy")
	else:
		faces = np.load("faces.npy")
	labels = np.load("labels.npy")
	isVSU = np.load("isVSU.npy")
	subjects = np.load("subjects.npy")
	return faces, labels, isVSU, subjects

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

	# Restrict to VSU
	faces = faces[np.nonzero(isVSU)]
	labels = labels[np.nonzero(isVSU)]
	subjects = subjects[np.nonzero(isVSU)]

	return faces, labels, isVSU, subjects

def trainNNRegression (faces, labels, subjects, foldIdx):
	faces -= np.tile(np.reshape(np.mean(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,FACE_SIZE,FACE_SIZE])
	faces /= np.tile(np.reshape(np.std(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,FACE_SIZE,FACE_SIZE])

	NUM_FOLDS = 4
	np.random.seed(0)  # Make sure the folds are always the same
	uniqueSubjects = np.random.permutation(np.unique(subjects))
	accuracies = []

	numSubjectsPerFold = int(np.ceil(len(uniqueSubjects) / float(NUM_FOLDS)))
	for i in range(NUM_FOLDS):
		if i != foldIdx:
			continue
		firstSubjectIdx = i*numSubjectsPerFold
		lastSubjectIdx = min((i+1)*numSubjectsPerFold, len(uniqueSubjects))
		testSubjects = uniqueSubjects[range(firstSubjectIdx, lastSubjectIdx)]

		idxs = np.nonzero(np.in1d(subjects, testSubjects) == False)[0]
		train_x = faces[idxs]
		train_x = np.reshape(train_x, train_x.shape + (1,))
		someLabels = labels[idxs]
		train_y = np.atleast_2d(someLabels).T.astype(np.float32)
		mx = np.mean(train_x, axis=0, keepdims=True)
		sx = np.std(train_x, axis=0, keepdims=True)
		sx[sx == 0] = 1
		# Normalize
		train_x = (train_x - mx) / sx

		# Shuffle training examples
		idxs = np.random.permutation(train_x.shape[0])
		train_x = train_x[idxs,:,:,:]
		train_y = train_y[idxs,:]

		idxs = np.nonzero(np.in1d(subjects, testSubjects) == True)[0]
		test_x = faces[idxs]
		test_x = np.reshape(test_x, test_x.shape + (1,))
		someLabels = labels[idxs]
		test_y = np.atleast_2d(someLabels).T.astype(np.float32)
		# Normalize
		test_x = (test_x - mx) / sx

		r = runNNRegression(train_x, train_y, test_x, test_y)
		print "Fold {}: {}".format(i, r)
		print ""

def trainNN (faces, labels, subjects, e):
	faces -= np.tile(np.reshape(np.mean(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,48,48])
	faces /= np.tile(np.reshape(np.std(faces, axis=(1,2)), (faces.shape[0], 1, 1)), [1,48,48])

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
		train_x = (train_x - mx) / sx

		# Shuffle training examples
		idxs = np.random.permutation(train_x.shape[0])
		train_x = train_x[idxs,:,:,:]
		train_y = train_y[idxs,:]

		idxs = np.nonzero(subjects == testSubject)[0]
		test_x = faces[idxs]
		test_x = np.reshape(test_x, test_x.shape + (1,))
		someLabels = labels[idxs]
		if len(np.unique(someLabels == e)) == 1:  # Don't bother if only 1 class in test set
			continue
		test_y = np.atleast_2d(someLabels != e).T
		test_y = np.hstack((test_y, 1 - test_y)).astype(np.float32)
		# Normalize
		test_x = (test_x - mx) / sx

		auc = runNNSimple(train_x, train_y, test_x, test_y)
		#auc = runNNRawPixels(train_x, train_y, test_x, test_y)
		print "{}: {}".format(testSubject, auc)
		print ""

def trainSVMRegression (filteredFaces, labels, subjects, masterK, C):
	accuracies = []

	NUM_FOLDS = 4
	np.random.seed(0)  # Make sure the folds are always the same
	uniqueSubjects = np.random.permutation(np.unique(subjects))
	accuracies = []

	numSubjectsPerFold = int(np.ceil(len(uniqueSubjects) / float(NUM_FOLDS)))
	for i in range(NUM_FOLDS):
		firstSubjectIdx = i*numSubjectsPerFold
		lastSubjectIdx = min((i+1)*numSubjectsPerFold, len(uniqueSubjects))
		testSubjects = uniqueSubjects[range(firstSubjectIdx, lastSubjectIdx)]

		trainIdxs = np.nonzero(np.in1d(subjects, testSubjects) == False)[0]
		someFilteredFacesTrain = filteredFaces[trainIdxs]
		someLabels = labels[trainIdxs]
		K = masterK[trainIdxs, :]
		K = K[:, trainIdxs]

		svms = []
		features = []
		for e in range(1, 5):
			y = someLabels == e
			svm = sklearn.svm.SVC(kernel="precomputed", C=C)
			svm.fit(K, y)
			svms.append(svm)
			yhat = svm.decision_function(K)
			features.append(yhat)
		features = np.array(features).T
		lr = sklearn.linear_model.LinearRegression()
		lr.fit(features, y)
		yhat = lr.predict(features)
		print np.corrcoef(y, yhat)[0,1]

		testIdxs = np.nonzero(np.in1d(subjects, testSubjects) == True)[0]
		someFilteredFaces = filteredFaces[testIdxs]
		K = masterK[testIdxs, :]
		K = K[:, trainIdxs]  # I.e., need trainIdxs dotted with testIdxs
		y = labels[testIdxs]
		features = []
		for j in range(len(svms)):
			yhat = svms[j].decision_function(K)
			features.append(yhat)
		features = np.array(features).T
		yhat = lr.predict(features)

		if len(np.unique(y)) > 1:
			r = np.corrcoef(y, yhat)[0,1]
		else:
			r = np.nan
		print "Fold {}: {}".format(i, r)
		accuracies.append(r)
	accuracies = np.array(accuracies)
	accuracies = accuracies[np.isfinite(accuracies)]
	print np.mean(accuracies), np.median(accuracies)

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

def weight_variable (shape, stddev = 0.1, wd = 0):
	values = stddev * np.random.randn(*shape).astype(np.float32)
	values = np.reshape(values, [ np.prod(shape[0:-1]), shape[-1] ])
	u,s,v = np.linalg.svd(values, full_matrices=False)
	u = np.reshape(u, shape)

	#var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	var = tf.Variable(u)

	weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
	tf.add_to_collection("losses", weight_decay)
	return var

def get_randomly_shifted (faces, cropSize):
	return faces
	#return faces[:, 2:46, 2:46, :]
	#diff = faces.shape[1] - cropSize
	#shiftedFaces = np.zeros((faces.shape[0], cropSize, cropSize, faces.shape[3]))
	#for i in range(faces.shape[0]):
	#	sx = np.random.randint(0, diff)
	#	sy = np.random.randint(0, diff)
	#	if np.random.random() < 0.5:
	#		shiftedFaces[i,:,:,:] = faces[i, sx:sx+cropSize, sy:sy+cropSize, :]
	#	else:
	#		shiftedFaces[i,:,:,:] = np.fliplr(faces[i, sx:sx+cropSize, sy:sy+cropSize, :])
	#return shiftedFaces

def bias_variable (shape, b=0.1):
	var = tf.constant(b, shape=shape)
	return tf.Variable(var)

def runNNRawPixels (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	BATCH_SIZE = 128
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		SIZE = 48
		x_image_resized = tf.image.resize_images(x_image, SIZE, SIZE)  # Downscale

		x_image_vec = tf.reshape(x_image_resized, [-1, SIZE*SIZE])
		W = weight_variable([SIZE*SIZE, train_y.shape[1]], stddev=0.001)
		b = bias_variable([train_y.shape[1]])

		yhat = tf.nn.softmax(tf.matmul(x_image_vec, W) + b)

		cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(yhat,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		#train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(total_loss)
		LEARNING_RATE = 0.05
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/5, 0.99, staircase=True)
		train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1).minimize(total_loss, global_step=batch)
		#train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			some_train_x = train_x[offset:offset+BATCH_SIZE, :, :, :]
			train_step.run({x_image: some_train_x, y_: train_y[offset:offset+BATCH_SIZE, :]})
			if i % 100 == 0:
				ll = total_loss.eval({x_image: train_x[:, :, :, :], y_: train_y})
				auc = sklearn.metrics.roc_auc_score(train_y[:,1], yhat.eval({x_image: train_x[:, :, :, :]})[:,1])
				print "Train LL={} AUC={}".format(ll, auc)

				ll = total_loss.eval({x_image: test_x[:, :, :, :], y_: test_y})
				auc = sklearn.metrics.roc_auc_score(test_y[:,1], yhat.eval({x_image: test_x[:, :, :, :]})[:,1])
				print "Test LL={} AUC={}".format(ll, auc)
		auc = sklearn.metrics.roc_auc_score(test_y[:,1], yhat.eval({x_image: test_x[:, :, :, :]})[:,1])
		session.close()
		return auc

def evalCorr (x_image, x, y_pred, y, keep_prob):
	yhat = []
	for i in range(int(np.ceil(x.shape[0] / float(BATCH_SIZE)))):
		idxs = range(i*BATCH_SIZE, min((i+1)*BATCH_SIZE, x.shape[0]))
		someYhat = y_pred.eval({x_image: x[idxs,:,:,:], keep_prob: 1.0}).squeeze()
		yhat += list(someYhat)
	return np.corrcoef(y.squeeze(), yhat)[0,1]

def runNNRegression (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		# Conv1
		NUM_FILTERS1 = 16
		W_conv1 = weight_variable([5, 5, 1, NUM_FILTERS1], stddev=0.01)
		b_conv1 = bias_variable([NUM_FILTERS1], b=1.)
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		# Pool
		h_pool1 = max_pool(h_conv1, 2)

		# Conv2
		NUM_FILTERS2 = 8
		W_conv2 = weight_variable([3, 3, NUM_FILTERS1, NUM_FILTERS2], stddev=0.01)
		b_conv2 = bias_variable([NUM_FILTERS2], b=1.)
		h_input2 = conv2d(h_pool1, W_conv2) + b_conv2
		h_conv2 = tf.nn.relu(h_input2)
		# Pool
		h_pool2 = max_pool(h_conv2, 2)

		# Vectorize
		h_pool2_reshaped = tf.reshape(h_pool2, [-1, (FACE_SIZE/4)*(FACE_SIZE/4)*NUM_FILTERS2])

		# Dropout
		keep_prob = tf.placeholder("float")
		h_pool1_drop = tf.nn.dropout(h_pool2_reshaped, keep_prob)

		# FC1
		W1 = weight_variable([ (FACE_SIZE/4)*(FACE_SIZE/4)*NUM_FILTERS2, train_y.shape[1] ], stddev=0.01, wd=1e-2)
		b1 = bias_variable([ train_y.shape[1] ], b=0.)
		fc1 = tf.matmul(h_pool1_drop, W1) + b1
		y_pred = 1. + 3*tf.sigmoid(fc1)
		#y_pred = tf.maximum(tf.constant(1.), tf.minimum(tf.constant(4.), fc1))

		l2 = tf.nn.l2_loss(y_ - y_pred, name='l2')
		tf.add_to_collection('losses', l2)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		#train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(total_loss)
		LEARNING_RATE = 0.0001
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/10, 0.9, staircase=True)
		#train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(total_loss, global_step=batch)
		train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			some_train_x = get_randomly_shifted(train_x[offset:offset+BATCH_SIZE, :, :, :], 44)
			some_train_y = train_y[offset:offset+BATCH_SIZE, :]
			train_step.run({x_image: some_train_x, y_: some_train_y, keep_prob: 0.25})
			if i % 100 == 0:
				ll = total_loss.eval({x_image: some_train_x[:, :, :, :], y_: some_train_y, keep_prob: 1.0})
				print "Train LL(some)={} r(all)={}".format(ll, evalCorr(x_image, train_x, y_pred, train_y, keep_prob))

				ll = total_loss.eval({x_image: test_x[:, :, :, :], y_: test_y, keep_prob: 1.0})
				print "Test LL(some)={} r(all)={}".format(ll, evalCorr(x_image, test_x, y_pred, test_y, keep_prob))

		r = evalCorr(x_image, test_x, y_pred, test_y, keep_prob)
		session.close()
		return r

def runNNSimple (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	BATCH_SIZE = 128
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		# Conv1
		NUM_FILTERS1 = 8
		W_conv1 = weight_variable([5, 5, 1, NUM_FILTERS1], stddev=0.01)
		b_conv1 = bias_variable([NUM_FILTERS1], b=1.)
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		# Pool
		h_pool1 = max_pool(h_conv1, 2)

		# Conv2
		NUM_FILTERS2 = 8
		W_conv2 = weight_variable([3, 3, NUM_FILTERS1, NUM_FILTERS2], stddev=0.01)
		b_conv2 = bias_variable([NUM_FILTERS2], b=1.)
		h_input2 = conv2d(h_pool1, W_conv2) + b_conv2
		h_conv2 = tf.nn.relu(h_input2)
		# Pool
		h_pool2 = max_pool(h_conv2, 2)

		# Vectorize
		h_pool2_reshaped = tf.reshape(h_pool2, [-1, 12*12*NUM_FILTERS2])

		# Dropout
		keep_prob = tf.placeholder("float")
		h_pool1_drop = tf.nn.dropout(h_pool2_reshaped, keep_prob)

		# FC1
		W1 = weight_variable([ 12*12*NUM_FILTERS2, train_y.shape[1] ], stddev=0.01, wd=1e-2)
		b1 = bias_variable([ train_y.shape[1] ], b=0.)
		fc1 = tf.matmul(h_pool1_drop, W1) + b1
		y_conv = tf.nn.softmax(fc1)

		cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		#train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(total_loss)
		LEARNING_RATE = 0.003
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/5, 0.99, staircase=True)
		#train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(total_loss, global_step=batch)
		train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			some_train_x = get_randomly_shifted(train_x[offset:offset+BATCH_SIZE, :, :, :], 44)
			train_step.run({x_image: some_train_x, y_: train_y[offset:offset+BATCH_SIZE, :], keep_prob: 0.9})
			if i % 100 == 0:
				#print h_pool1.eval({x_image: train_x[0:1, 2:46, 2:46, :], keep_prob: 1.0})
				#print h_input2.eval({x_image: train_x[0:1, 2:46, 2:46, :], keep_prob: 1.0})
				ll = total_loss.eval({x_image: train_x[:, :, :, :], y_: train_y, keep_prob: 1.0})
				auc = sklearn.metrics.roc_auc_score(train_y[:,1], y_conv.eval({x_image: train_x[:, :, :, :], keep_prob: 1.0})[:,1])
				print "Train LL={} AUC={}".format(ll, auc)

				ll = total_loss.eval({x_image: test_x[:, :, :, :], y_: test_y, keep_prob: 1.0})
				auc = sklearn.metrics.roc_auc_score(test_y[:,1], y_conv.eval({x_image: test_x[:, :, :, :], keep_prob: 1.0})[:,1])
				print "Test LL={} AUC={}".format(ll, auc)

				#plt.imshow(np.reshape(h_conv1.eval({x_image: train_x[0:1, 2:46, 2:46,:]})[0,:,:,0], [44,44]))
				#plt.show()
				#plt.imshow(np.reshape(h_pool1.eval({x_image: train_x[0:1, 2:46, 2:46,:]})[0,:,:,0], [11,11]))
				#plt.show()
				#plt.imshow(np.reshape(W2.eval()[0:121,0], [ 11,11 ]))
				#plt.show()
		auc = sklearn.metrics.roc_auc_score(test_y[:,1], y_conv.eval({x_image: test_x[:, :, :, :], keep_prob: 1.0})[:,1])
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
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS/5, 0.99, staircase=True)
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
	C = float(sys.argv[1])

	if 'faces' not in globals():
		#faces, labels, isVSU, subjects = getData()
		faces, labels, isVSU, subjects = getDataFast()
	
		filterBank = gabor.makeGaborFilterBank(faces.shape[-1])
		filterBankF = np.fft.fft2(filterBank)
		filteredFaces = filterFaces(faces, filterBankF)
		
		masterK = filteredFaces.dot(filteredFaces.T)

	#for e in [ 1, 2, 3, 4 ]:  # Engagement label
	#	print "E={}".format(e)
	#	#trainSVM(filteredFaces, labels, subjects, e)
	#	trainNN(faces, labels, subjects, e)

	#trainNNRegression(faces, labels, subjects, foldIdx)
	trainSVMRegression(filteredFaces, labels, subjects, masterK, C)
