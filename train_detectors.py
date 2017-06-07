import cPickle
import skimage.transform
import sys
import tensorflow as tf
import sklearn
import sklearn.svm
import sklearn.kernel_ridge
import gabor
import matplotlib.pyplot as plt
import numpy as np
import pandas
from skimage.transform import resize

FACE_SIZE = 48
NUM_EPOCHS = 100
BATCH_SIZE = 64

SUBJECTS_IN_FOLDS = np.array([
    "FC22", "FC20", "FC18", "FC21", "FC24",  # Fold 1
    "FC25", "FO19", "FO21", "FO22", "FO20",  # Fold 2
    "FO25", "FO24", "FW22", "FW18", "FW21",  # Fold 3
    "FW24", "FW23", "MC08", "MO07", "MW07"   # Fold 4
])
#NUM_FOLDS = len(SUBJECTS_IN_FOLDS)
NUM_FOLDS = 4
NUM_SUBJECTS_PER_FOLD = len(SUBJECTS_IN_FOLDS)/NUM_FOLDS

def resizeFaces (faces, newSize):
    newFaces = np.zeros((faces.shape[0], newSize, newSize), dtype=np.float32)
    for i in range(faces.shape[0]):
        newFaces[i,:,:] = resize(faces[i,:,:], (newSize, newSize), preserve_range=True)
    return newFaces

def conv2d (x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool (x, size=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def normalize (faces):
    normedFaces = np.zeros(faces.shape)
    for i in range(faces.shape[0]):
        normedFaces[i,:,:] = faces[i,:,:] - np.mean(faces[i,:,:])
        normedFaces[i,:,:] /= np.linalg.norm(normedFaces[i,:,:])
    return normedFaces

def filterFaces (faces, filterBankF):
    CHUNK = 1000
    results = []
    for i in range(int(np.ceil(faces.shape[0]/float(CHUNK)))):
        print "Chunk {}".format(i)
        idxs = range(i*CHUNK, min((i+1)*CHUNK, faces.shape[0]))
        someResults = filterSomeFaces(normalize(faces[idxs, :, :]), filterBankF)
        results.append(someResults)
    return np.vstack(results)

def filterSomeFaces (faces, filterBankF):
    facesF = np.tile(np.fft.fft2(faces).reshape((faces.shape[0], 1, faces.shape[1], faces.shape[2])), (1, len(filterBankF), 1, 1))
    kernelsF = np.tile(filterBankF.reshape((1,) + filterBankF.shape), (faces.shape[0], 1, 1, 1))
    #resultsF = facesF * np.conjugate(kernelsF)
    resultsF = facesF * kernelsF
    del facesF
    del kernelsF
    results = np.abs(np.fft.ifft2(resultsF))
    del resultsF
    results /= np.linalg.norm(results, axis=(2,3), keepdims=True)
    results = np.reshape(results.astype(np.float32), (faces.shape[0], len(filterBankF)*faces.shape[1]*faces.shape[2]))
    return results

def getDataFast ():
    if FACE_SIZE == 36:
        faces = np.load("faces36.npy")
    else:
        faces = np.load("faces.npy")
    labels = np.load("labels.npy")
    subjects = np.load("subjects.npy")
    return faces, labels, subjects

def getData ():
    allFrameFilenames = cPickle.load(open("filenames.pkl", "rb"))
    allFrameFilenames = np.array([ filename[0:-1] for filename in allFrameFilenames ])  # Remove '\n'
    allFaces = np.load("thefaces.npy")
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
    accuracies = []
    for i in range(NUM_FOLDS):
        if i != foldIdx:
            continue
        firstSubjectIdx = i*NUM_SUBJECTS_PER_FOLD
        lastSubjectIdx = min((i+1)*NUM_SUBJECTS_PER_FOLD, len(SUBJECTS_IN_FOLDS))
        testSubjects = SUBJECTS_IN_FOLDS[range(firstSubjectIdx, lastSubjectIdx)]

        idxs = np.nonzero(np.in1d(subjects, testSubjects) == False)[0]
        train_x = faces[idxs]
        train_x = np.reshape(train_x, train_x.shape + (1,))
        someLabels = labels[idxs]
        train_y = np.atleast_2d(someLabels).T.astype(np.float32)
        
        # Normalize
        #mx = np.mean(train_x, axis=0, keepdims=True)
        #sx = np.std(train_x, axis=0, keepdims=True)
        #sx[sx == 0] = 1
        #train_x = (train_x - mx) / sx

        # Shuffle training examples
        idxs = np.random.permutation(train_x.shape[0])
        train_x = train_x[idxs,:]
        train_y = train_y[idxs,:]

        idxs = np.nonzero(np.in1d(subjects, testSubjects) == True)[0]
        test_x = faces[idxs]
        test_x = np.reshape(test_x, test_x.shape + (1,))
        someLabels = labels[idxs]
        test_y = np.atleast_2d(someLabels).T.astype(np.float32)
        # Normalize
        #test_x = (test_x - mx) / sx

        #r = runFCRegression(train_x, train_y, test_x, test_y)
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

def trainLinearRegressor (features, y, subjects):
    alphas = 10. ** np.arange(-2, +3)
    uniqueSubjects, subjectIdxs = np.unique(subjects, return_inverse=True)
    highestAccuracy = - float('inf')
    NUM_MINI_FOLDS = 4
    for alpha in alphas:  # For each regularization value
        accuracies = []
        for i in range(NUM_MINI_FOLDS):  # For each test subject
            testIdxs = np.nonzero(subjectIdxs % NUM_MINI_FOLDS == i)[0]
            trainIdxs = np.nonzero(subjectIdxs % NUM_MINI_FOLDS != i)[0]

            lr = sklearn.linear_model.Ridge(normalize=True, fit_intercept=True, alpha=alpha)
            lr.fit(features[trainIdxs,:], y[trainIdxs])
            accuracy = np.corrcoef(lr.predict(features[testIdxs]), y[testIdxs])[0,1]
            accuracies.append(accuracy)
        if np.mean(accuracies) > highestAccuracy:
            highestAccuracy = np.mean(accuracies)
            bestAlpha = alpha
            print "best alpha = {}".format(bestAlpha)
    lr = sklearn.linear_model.Ridge(normalize=True, fit_intercept=True, alpha=bestAlpha)
    lr.fit(features, y)
    return lr

def trainOneSVM (masterK, y, subjects):
    Cs = 1. / np.array([ 0.1, 0.5, 2.5, 12.5, 62.5, 312.5 ])
    #Cs = 10. ** np.arange(-5, +6)/2.
    uniqueSubjects, subjectIdxs = np.unique(subjects, return_inverse=True)
    highestAccuracy = - float('inf')
    NUM_MINI_FOLDS = 4
    for C in Cs:  # For each regularization value
        #print "C={}".format(C)
        accuracies = []
        for i in range(NUM_MINI_FOLDS):  # For each test subject
            testIdxs = np.nonzero(subjectIdxs % NUM_MINI_FOLDS == i)[0]
            trainIdxs = np.nonzero(subjectIdxs % NUM_MINI_FOLDS != i)[0]
            if len(np.unique(y[testIdxs])) > 1:
                K = masterK[trainIdxs, :]
                K = K[:, trainIdxs]
                svm = sklearn.svm.SVC(kernel="precomputed", C=C)
                svm.fit(K, y[trainIdxs])

                K = masterK[testIdxs, :]
                K = K[:, trainIdxs]  # I.e., need trainIdxs dotted with testIdxs
                accuracy = sklearn.metrics.roc_auc_score(y[testIdxs], svm.decision_function(K))
                #print accuracy
                accuracies.append(accuracy)
        if np.mean(accuracies) > highestAccuracy:
            highestAccuracy = np.mean(accuracies)
            bestC = C
    svm = sklearn.svm.SVC(kernel="precomputed", C=bestC)
    svm.fit(masterK, y)
    return svm

def trainSVMRegression (labels, subjects, masterK, alpha):
    accuracies = []
    for i in range(NUM_FOLDS):
        firstSubjectIdx = i*NUM_SUBJECTS_PER_FOLD
        lastSubjectIdx = min((i+1)*NUM_SUBJECTS_PER_FOLD, len(SUBJECTS_IN_FOLDS))
        testSubjects = SUBJECTS_IN_FOLDS[range(firstSubjectIdx, lastSubjectIdx)]

        trainIdxs = np.nonzero(np.in1d(subjects, testSubjects) == False)[0]
        testIdxs = np.nonzero(np.in1d(subjects, testSubjects) == True)[0]
        trainK = masterK[trainIdxs,:]
        trainK = trainK[:,trainIdxs]
        testK = masterK[testIdxs,:]
        testK = testK[:,trainIdxs]
        trainY = labels[trainIdxs]
        testY = labels[testIdxs]

        lr = sklearn.kernel_ridge.KernelRidge(kernel="precomputed", alpha=alpha)
        lr.fit(trainK, trainY)
        yhat = lr.predict(testK)

        if len(np.unique(testY)) > 1:
            r = np.corrcoef(testY, yhat)[0,1]
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
    if values.shape[-1] < values.shape[0]:
        u,s,v = np.linalg.svd(values, full_matrices=False)
    else:
        u = values
    u = np.reshape(u, shape)
    var = tf.Variable(u)
    
    #var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
    tf.add_to_collection("losses", weight_decay)
    return var

def getRandomlyAltered (faces, replication):
    DIFF = 4
    alteredFaces = []
    for r in range(replication):
        for i in range(faces.shape[0]):
            sx = np.random.randint(0, DIFF)
            sy = np.random.randint(0, DIFF)
            sw = np.random.randint(0, DIFF)
            sh = np.random.randint(0, DIFF)

            face = faces[i,sy:FACE_SIZE-sh,sx:FACE_SIZE-sw].squeeze()
            minval = np.min(face, keepdims=True)
            maxval = np.max(face, keepdims=True)
            face = (face - minval) / (maxval - minval)

            if np.random.random() < 0.5:  # Flip left-right
                face = face[:,::-1]
            face = skimage.transform.resize(face, [FACE_SIZE, FACE_SIZE])

            face = face * (maxval - minval) + minval

            alteredFaces.append(face)
            #plt.imshow(np.hstack((face, faces[i,:,:].squeeze()))), plt.show()
    return np.resize(alteredFaces, [ replication*faces.shape[0], FACE_SIZE, FACE_SIZE, 1 ])

def bias_variable (shape, b=0.1):
    var = tf.constant(b, shape=shape)
    return tf.Variable(var)

def runNNRawPixels (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
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

def evalCorr (x_image, x, y_pred, y, keep_prob = None):
    yhat = []
    for i in range(int(np.ceil(x.shape[0] / float(BATCH_SIZE)))):
        idxs = range(i*BATCH_SIZE, min((i+1)*BATCH_SIZE, x.shape[0]))
        if len(x_image.get_shape()) > 2:
            if keep_prob == None:
                someYhat = y_pred.eval({x_image: x[idxs,:,:,:]}).squeeze()
            else:
                someYhat = y_pred.eval({x_image: x[idxs,:,:,:], keep_prob: 1.0}).squeeze()
        else:
            someYhat = y_pred.eval({x_image: x[idxs,:], keep_prob: 1.0}).squeeze()
        yhat += list(someYhat)
    return np.corrcoef(y.squeeze(), yhat)[0,1], yhat

def spatial_lrn (tensor, radius=5, bias=1.0, alpha=1.0, beta=0.5):
    squared = tf.square(tensor)
    in_channels = tensor.get_shape().as_list()[3]
    kernel = tf.constant(1.0, shape=[radius, radius, in_channels, 1])
    squared_sum = tf.nn.depthwise_conv2d(squared,
                                         kernel,
                                         [1, 1, 1, 1],
                                         padding='SAME')
    bias = tf.constant(bias, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    return tensor / tf.pow((bias + alpha * squared_sum), beta)

def runNNRegression (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
    with tf.Graph().as_default():
        session = tf.InteractiveSession()

        x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
        y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

        # LRN
        #x_normed = spatial_lrn(x_image)
        x_normed = x_image

        # Conv1
        NUM_FILTERS1 = 16
        W_conv1 = weight_variable([3, 3, 1, NUM_FILTERS1], stddev=0.01)
        b_conv1 = bias_variable([NUM_FILTERS1], b=0.1)
        h_conv1 = tf.nn.relu(conv2d(x_normed, W_conv1) + b_conv1)
        h_pool1 = max_pool(h_conv1, 2)

        # Conv2
        NUM_FILTERS2 = 16
        W_conv2 = weight_variable([3, 3, NUM_FILTERS1, NUM_FILTERS2], stddev=0.01)
        b_conv2 = bias_variable([NUM_FILTERS2], b=0.1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool(h_conv2, 2)

        # Conv3
        NUM_FILTERS3 = 8
        W_conv3 = weight_variable([3, 3, NUM_FILTERS2, NUM_FILTERS3], stddev=0.01)
        b_conv3 = bias_variable([NUM_FILTERS3], b=0.1)
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool(h_conv3, 2)

        # Vectorize
        h_reshaped = tf.reshape(h_pool3, [-1, 5*5*NUM_FILTERS3])

        # FC1
        W1 = weight_variable([ 5*5*NUM_FILTERS3, 1 ], stddev=0.1, wd=1e-3)
        b1 = bias_variable([ 1 ], b=1.)
        fc1 = tf.matmul(h_reshaped, W1) + b1

        y_pred = fc1
        #y_pred = tf.maximum(tf.constant(1.), tf.minimum(tf.constant(4.), fc1))

        l2 = tf.nn.l2_loss(y_ - y_pred, name='l2')
        tf.add_to_collection('losses', l2)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        numBatches = int(np.ceil(train_x.shape[0] / float(BATCH_SIZE)))
        LEARNING_RATE = 0.001
        batch = tf.Variable(0)
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

        session.run(tf.global_variables_initializer())
        for i in range(numEpochs):
            if i > 0 and i % 2 == 0:
                print "Epoch {}".format(i)

                ll = total_loss.eval({x_image: train_x, y_: train_y})
                print "Train LL(some)={} r(all)={}".format(ll, evalCorr(x_image, train_x, y_pred, train_y)[0])

                ll = total_loss.eval({x_image: test_x, y_: test_y})
                print "Test LL(some)={} r(all)={}".format(ll, evalCorr(x_image, test_x, y_pred, test_y)[0])

                np.save("W1.npy", W1.eval())
                np.save("b1.npy", b1.eval())
                np.save("W_conv1.npy", W_conv1.eval())
                np.save("b_conv1.npy", b_conv1.eval())
                np.save("W_conv2.npy", W_conv2.eval())
                np.save("b_conv2.npy", b_conv2.eval())
                np.save("W_conv3.npy", W_conv3.eval())
                np.save("b_conv3.npy", b_conv3.eval())
            for j in range(numBatches):
                offset = j*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
                some_train_x = train_x[offset:offset+BATCH_SIZE, :, :, :]
                some_train_y = train_y[offset:offset+BATCH_SIZE, :]
                train_step.run({x_image: some_train_x, y_: some_train_y})
                if j % 10 == 0:
                    #print tf.concat(1, [ y_, y_pred ]).eval({x_image: some_train_x, y_: some_train_y})
                    print total_loss.eval({x_image: some_train_x, y_: some_train_y}), np.linalg.norm(some_train_y - np.mean(some_train_y))**2 / 2
                    #print np.corrcoef(h_conv1.eval({x_image: some_train_x, y_: some_train_y}).reshape(24*24,-1))

        r = evalCorr(x_image, test_x, y_pred, test_y)
        session.close()
        return r

def runNNSimple (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
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
            some_train_x = getRandomlyAltered(train_x[offset:offset+BATCH_SIZE, :, :, :], 44)
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

def runFCRegression (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
    #REPLICATION = 2
    #train_x = np.vstack((train_x, getRandomlyAltered(train_x, REPLICATION)))
    #train_y = np.vstack((train_y, np.tile(train_y, (REPLICATION,1))))

    #train_x = train_x.reshape(train_x.shape[0], np.prod(train_x.shape[1:]))
    #test_x = test_x.reshape(test_x.shape[0], np.prod(test_x.shape[1:]))

    with tf.Graph().as_default():
        session = tf.InteractiveSession()

        x_image = tf.placeholder("float", shape=[None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
        y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

        # Conv1
        NUM_FILTERS1 = 32
        W_conv1 = weight_variable([9, 9, 1, NUM_FILTERS1], stddev=0.1)
        b_conv1 = bias_variable([NUM_FILTERS1], b=1.)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # Pool
        h_pool1 = max_pool(h_conv1, 2)

        # Conv1
        NUM_FILTERS2 = 16
        W_conv2 = weight_variable([9, 9, NUM_FILTERS1, NUM_FILTERS2], stddev=0.1)
        b_conv2 = bias_variable([NUM_FILTERS2], b=1.)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # Pool
        h_pool2 = max_pool(h_conv2, 2)
        h_pool2_reshaped = tf.reshape(h_pool2, [ -1, NUM_FILTERS2*FACE_SIZE/4*FACE_SIZE/4 ])

        W1 = weight_variable([NUM_FILTERS2*FACE_SIZE/4*FACE_SIZE/4, 1], stddev=0.01)
        b1 = bias_variable([1])
        y_pred = tf.matmul(h_pool2_reshaped, W1) + b1

        l2 = tf.nn.l2_loss(y_ - y_pred, name='l2')
        tf.add_to_collection('losses', l2)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        LEARNING_RATE = 1e-6
        batch = tf.Variable(0)
        numBatches = int(np.ceil(train_x.shape[0] / float(BATCH_SIZE)))
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch, NUM_EPOCHS*numBatches/10, 0.95, staircase=False)
        train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.2).minimize(total_loss, global_step=batch)

        session.run(tf.initialize_all_variables())
        for i in range(numEpochs):
            for j in range(numBatches):
                offset = j*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
                some_train_x = train_x[offset:offset+BATCH_SIZE, :]
                some_train_y = train_y[offset:offset+BATCH_SIZE, :]
                train_step.run({x_image: some_train_x, y_: some_train_y})
                ll = total_loss.eval({x_image: some_train_x, y_: some_train_y})
                print ll

            if i % 5 == 0:
                print "Epoch {} (lr={})".format(i, learning_rate.eval())

                ll = total_loss.eval({x_image: train_x, y_: train_y})
                r, _ = evalCorr(x_image, train_x, y_pred, train_y)
                print "Train LL(some)={} r(all)={}".format(ll, r)

                ll = total_loss.eval({x_image: test_x, y_: test_y})
                r, yhat = evalCorr(x_image, test_x, y_pred, test_y)
                print "Test LL(some)={} r(all)={}".format(ll, r)

                #print np.corrcoef(W1.eval().T)

                plt.imshow(np.hstack([ np.pad(np.reshape(W_conv1.eval()[:,:,:,idx], [ 9, 9 ]), 2, mode='constant') for idx in range(NUM_FILTERS1) ]), cmap='gray'), plt.show()

                plt.imshow(np.vstack([ \
                             np.hstack([ np.pad(np.reshape(W_conv2.eval()[:,:,idx1,idx2], [ 9, 9 ]), 2, mode='constant') for idx1 in range(NUM_FILTERS1) ]) \
                         for idx2 in range(NUM_FILTERS2) ]), cmap='gray'), plt.show()
            
                filters = np.reshape(W1.eval(), [ NUM_FILTERS2, FACE_SIZE/4, FACE_SIZE/4 ])
                plt.imshow(np.hstack([ filters[idx, :,:] for idx in range(NUM_FILTERS2) ]), cmap='gray'), plt.show()

                # Show mistakes
                #idxs = np.argsort((yhat - test_y.squeeze()) ** 2)[-10:]
                #print [ yhat[idx] for idx in idxs ]
                #print test_y[idxs].T
                #plt.imshow(np.hstack([ np.reshape(test_x[idx,:], (FACE_SIZE, FACE_SIZE)) for idx in idxs ]), cmap='gray')
                plt.show()

        r, _ = evalCorr(x_image, test_x, y_pred, test_y)
        session.close()
        return r

if __name__ == "__main__":
    foldIdx = int(sys.argv[1])

    if 'faces' not in globals():
        #faces, labels, isVSU, subjects = getData()
        faces, labels, subjects = getDataFast()

        FILTER = False
        if FILTER:
            filterBank = gabor.makeGaborFilterBank(faces.shape[-1])
            filterBankF = np.fft.fft2(filterBank)
            filteredFaces = filterFaces(faces, filterBankF)
            masterK = filteredFaces.dot(filteredFaces.T)

    trainNNRegression(normalize(faces), labels, subjects, foldIdx)

    #alphas = 10. ** np.arange(-2, +3)
    #for alpha in alphas:
    #    print alpha
    #    trainSVMRegression(labels, subjects, masterK, alpha)
