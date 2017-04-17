import matplotlib.pyplot as plt
import numpy as np

labels = np.load('labels.npy')
faces = np.load('faces.npy')
faces = (faces - np.mean(faces, axis=0, keepdims=True)) / np.std(faces, axis=0, keepdims=True)
for label in range(1, 5):
    plt.imshow(np.mean(faces[labels == label], axis=0), cmap='gray'), plt.show()
