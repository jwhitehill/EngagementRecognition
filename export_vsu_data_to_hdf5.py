import h5py
import numpy as np
from common import SUBJECTS_IN_FOLDS, NUM_FOLDS, NUM_SUBJECTS_PER_FOLD, normalize

foldIdx = 3
faces = normalize(np.load('faces.npy'))
faces = faces.reshape(faces.shape[0], 1, faces.shape[1], faces.shape[2])
labels = np.load("labels.npy")
subjects = np.load("subjects.npy")

firstSubjectIdx = foldIdx*NUM_SUBJECTS_PER_FOLD
lastSubjectIdx = min((foldIdx+1)*NUM_SUBJECTS_PER_FOLD, len(SUBJECTS_IN_FOLDS))
testSubjects = SUBJECTS_IN_FOLDS[range(firstSubjectIdx, lastSubjectIdx)]

trainIdxs = np.nonzero(np.in1d(subjects, testSubjects) == False)[0]
trainIdxs = np.random.permutation(trainIdxs)
testIdxs = np.nonzero(np.in1d(subjects, testSubjects) == True)[0]

trainFaces = faces[trainIdxs,:,:,:]
trainLabels = labels[trainIdxs]
testFaces = faces[testIdxs,:,:,:]
testLabels = labels[testIdxs]

for (filename, someFaces, someLabels) in [ ("train", trainFaces, trainLabels), ("test", testFaces, testLabels) ]:
    with h5py.File(filename + str(foldIdx) + ".h5", 'w') as f:
        f['data'] = someFaces
        f['label'] = someLabels
    with open(filename + str(foldIdx) + ".txt", 'w') as f:
        f.write("examples/EngagementRecognition/" + filename + str(foldIdx) + ".h5")
