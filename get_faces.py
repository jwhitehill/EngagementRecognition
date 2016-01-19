import numpy as np
import cPickle
import StringIO
import matplotlib.pyplot as plt
import scipy.misc
import re

filenameRE = re.compile(r'\/Users\/jaw291\/Projects\/EngagementRecognition\/\/([^\t]*)')  # "\t*" -- I accidentally included extra tabs
faceRE = re.compile(r'\(Face (\d+) (\d+) (\d+) (\d+)\) .*')

def decodeFace (lines):
	rows = []
	fixedLines = []
	for line in lines:
		line = line.replace("[", "")
		line = line.replace("]", "")
		line = line.replace(",", "")
		line = line.replace(";", "")
		fixedLines.append(line)
	c = StringIO.StringIO("".join(fixedLines))
	return np.loadtxt(c)

if __name__ == "__main__":
	faces = {}
	f = open('faces.txt', 'r')
	FACE_SIZE = 48
	idx = 0
	while True:
		line = f.readline()
		if line == "":
			break
		m = filenameRE.match(line)
		if m != None:
			partialFilename = m.group(1)
			line = f.readline()
			if line[0] == '[':
				lines = [ line ]
				for i in range(FACE_SIZE - 1):  # We already read one line
					lines.append(f.readline())
				face = decodeFace(lines)
				#plt.imshow(face)
				#plt.show()
				faces[partialFilename] = face
				idx += 1
				if idx % 100 == 0:
					print idx
			else:  # Not an image, so backtrack 1 line
				f.seek(-len(line),1)
	cPickle.dump((faces.keys(), np.array(faces.values()).astype(np.float32)), open("faces.pkl", "wb"))
