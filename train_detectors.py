import cPickle
import sklearn
import sklearn.svm
import gabor
import matplotlib.pyplot as plt
import numpy as np
import pandas

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
		
def train (filteredFaces, labels, subjects, e):
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
	accuracies = accuracies[np.isfinite(accuracies)]
	print np.mean(accuracies), np.median(accuracies)

if __name__ == "__main__":
	faces, labels, isVSU, subjects = getData()
	
	# Restrict to VSU
	faces = faces[np.nonzero(isVSU)]
	labels = labels[np.nonzero(isVSU)]
	subjects = subjects[np.nonzero(isVSU)]

	filterBank = gabor.makeGaborFilterBank(faces.shape[-1])
	filterBankF = np.fft.fft2(filterBank)
	filteredFaces = filterFaces(faces, filterBankF)

	#for e in [ 1, 2, 3, 4 ]:  # Engagement label
	for e in [ 1 ]:  # Engagement label
		train(filteredFaces, labels, subjects, e)
