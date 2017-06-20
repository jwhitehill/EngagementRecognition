import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn

FACE_SIZE = 48

SUBJECTS_IN_FOLDS = np.array([
    "FC22", "FC20", "FC18", "FC21", "FC24",  # Fold 1
    "FC25", "FO19", "FO21", "FO22", "FO20",  # Fold 2
    "FO25", "FO24", "FW22", "FW18", "FW21",  # Fold 3
    "FW24", "FW23", "MC08", "MO07", "MW07"   # Fold 4
])
NUM_FOLDS = 4
NUM_SUBJECTS_PER_FOLD = len(SUBJECTS_IN_FOLDS)/NUM_FOLDS

def conv2d (x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool (x, size=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def normalize (faces):
    normedFaces = np.zeros(faces.shape)
    for i in range(faces.shape[0]):
        normedFaces[i,:,:] = faces[i,:,:] - np.mean(faces[i,:,:])
        normedFaces[i,:,:] /= np.linalg.norm(normedFaces[i,:,:])
    return normedFaces

def getDataFast ():
    if FACE_SIZE == 36:
        faces = np.load("faces36.npy")
    else:
        faces = np.load("faces.npy")
    labels = np.load("labels.npy")
    subjects = np.load("subjects.npy")
    return faces, labels, subjects

def bias_variable (shape, b=0.1):
    var = tf.constant(b, shape=shape)
    return tf.Variable(var)

def weight_variable (shape, stddev = 0.1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    return var

def V (X):
    numRows = X.get_shape().as_list()[1]
    count, _, _, _ = tf.nn.sufficient_statistics(X, [0])
    _, sigma = tf.nn.moments(X, [0])
    ALPHA = 1e-1  # For regularization
    return count * ((1 - ALPHA) * sigma + ALPHA * tf.eye(numRows))

def W (X1, X0):
    return V(X1) + V(X0)
    
def J (X1, X0):
    W_ = W(X1, X0)
    meanX1 = tf.reduce_mean(X1, axis=0, keep_dims=True)
    meanX0 = tf.reduce_mean(X0, axis=0, keep_dims=True)
    meanDiff = meanX1 - meanX0
    meanDiffT = tf.transpose(meanDiff)
    B_ = tf.matmul(meanDiffT, meanDiff)
    p = tf.matrix_solve(W_, meanDiffT)
    return tf.matmul(tf.transpose(p), tf.matmul(B_, p)) / \
           tf.matmul(tf.transpose(p), tf.matmul(W_, p))

def DDD (X1a, X0a, X1b, X0b):
    with tf.Graph().as_default():
        session = tf.InteractiveSession()

        x1a = tf.placeholder("float", shape=[None, X1a.shape[1], X1a.shape[2], X1a.shape[3]])
        x0a = tf.placeholder("float", shape=[None, X0a.shape[1], X0a.shape[2], X0a.shape[3]])
        x1b = tf.placeholder("float", shape=[None, X1b.shape[1], X1b.shape[2], X1b.shape[3]])
        x0b = tf.placeholder("float", shape=[None, X0b.shape[1], X0a.shape[2], X0b.shape[3]])

        # Conv1
        NUM_FILTERS1 = 4
        W_conv1 = weight_variable([5, 5, 1, NUM_FILTERS1], stddev=0.01)
        b_conv1 = bias_variable([NUM_FILTERS1], b=0.1)
        f1a = tf.reshape(max_pool(tf.nn.relu(conv2d(x1a, W_conv1) + b_conv1), 2), [ -1, 4*4*NUM_FILTERS1 ])
        f0a = tf.reshape(max_pool(tf.nn.relu(conv2d(x0a, W_conv1) + b_conv1), 2), [ -1, 4*4*NUM_FILTERS1 ])
        f1b = tf.reshape(max_pool(tf.nn.relu(conv2d(x1b, W_conv1) + b_conv1), 2), [ -1, 4*4*NUM_FILTERS1 ])
        f0b = tf.reshape(max_pool(tf.nn.relu(conv2d(x0b, W_conv1) + b_conv1), 2), [ -1, 4*4*NUM_FILTERS1 ])

        loss_a = J(f1a, f0a)
        loss_b = J(f1b, f0b)
        loss = tf.log(loss_b) - tf.log(loss_a)

        LEARNING_RATE = 0.001
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        session.run(tf.global_variables_initializer())
        NUM_EPOCHS = 500
        BATCH_SIZE = 256
        for i in range(NUM_EPOCHS):
            def pickSome (M):
                return M[np.random.permutation(M.shape[0])[0:BATCH_SIZE], :,:,:]
            some_x1a = pickSome(X1a)
            some_x0a = pickSome(X0a)
            some_x1b = pickSome(X1b)
            some_x0b = pickSome(X0b)

            train_step.run({x1a: some_x1a, x0a: some_x0a, x1b: some_x1b, x0b: some_x0b})
            if i % 1 == 0:
                myDict = {x1a: X1a, x0a: X0a, x1b: X1b, x0b: X0b}
                la = loss_a.eval(myDict)
                lb = loss_b.eval(myDict)
                l = loss.eval(myDict)
                print "Train LL={} loss_a={} loss_b={}".format(l, la, lb)
            if i % 5 == 0:
                plt.imshow(np.hstack([ np.pad(np.reshape(W_conv1.eval()[:,:,:,idx], [ 5, 5 ]), 2, mode='constant') for idx in range(NUM_FILTERS1) ]), cmap='gray'), plt.show()

        session.close()

def demoOnFaces ():
    if 'faces' not in globals():
        faces, labels, subjects = getDataFast()
        # Randomize
        idxs = np.random.permutation(faces.shape[0])
        faces = faces[idxs,:,:]
        labels = labels[idxs]
        subjects = subjects[idxs]

    X = faces.reshape(faces.shape + (1,))
    # Discriminate between C and not-C conditions
    conditions = np.array([ code[1] for code in subjects ])
    Ya = (conditions == 'C').astype(np.int32)
    Yb = (labels > 3).astype(np.int32)
    X1a = X[np.nonzero(Ya == 1)[0], :,:,:]
    X0a = X[np.nonzero(Ya == 0)[0], :,:,:]
    X1b = X[np.nonzero(Yb == 1)[0], :,:,:]
    X0b = X[np.nonzero(Yb == 0)[0], :,:,:]
    DDD(X1a, X0a, X1b, X0b)

def demoOnStripes ():
    N = 250
    NOISE = 0.5
    P = 16
       
    X = NOISE * np.random.rand(N, P*P)
    featureIdxs = np.reshape(range(P*P), [ P, P ]);
    lineIdxs = range(P/8, 7*P/8)
    
    # Randomly create labels (Ya) for task a -- modify data (X) accordingly
    Ya = np.random.random(N) > 0.5
    for i in range(N):
        if Ya[i] == 1:
            idxs = featureIdxs[int(np.ceil(np.random.random() * P/4)) + P/8, lineIdxs]
        else:
            idxs = featureIdxs[int(np.ceil(np.random.random() * P/4)) + 5*P/8, lineIdxs]
        X[i, idxs] += 1
    
    # Randomly create labels (Yb) for task b -- modify data (X) accordingly
    Yb = np.random.random(N) > 0.5
    for i in range(N):
        if Yb[i] == 1:
            idxs = featureIdxs[lineIdxs, int(np.ceil(np.random.random() * P/4)) + P/8]
        else:
            idxs = featureIdxs[lineIdxs, int(np.ceil(np.random.random() * P/4)) + 5*P/8]
        X[i, idxs] += 1
    
    # Now, partition into X1a and X0a for task a, and X1b an X0b for task b
    X1a = X[Ya==1, :]
    X0a = X[Ya==0, :]
    X1b = X[Yb==1, :]
    X0b = X[Yb==0, :]

    # Add extra dimension
    X1a = X1a.reshape((-1,P,P,1))
    X0a = X0a.reshape((-1,P,P,1))
    X1b = X1b.reshape((-1,P,P,1))
    X0b = X0b.reshape((-1,P,P,1))

    # Run DDD
    DDD(X1a, X0a, X1b, X0b)

if __name__ == "__main__":
    #demoOnFaces()
    demoOnStripes()
