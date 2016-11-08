import re
import numpy as np
import pandas
import os

theRE = re.compile(r'^Spring2011\/frames\/([^\/]+)\/.*')

def getSubject (filename):
    match = theRE.match(filename)
    return match.group(1)

def getLabels ():
    labels = []
    d = pandas.io.parsers.read_csv('labels.txt', sep='\t')
    e = d.groupby("frame_filename")
    frameToLabelMap = {}
    for frameFilename in e.indices.keys():
        if "UCSD" in frameFilename:
            continue
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
        frameToLabelMap[frameFilename] = avgLabel
    return frameToLabelMap
    
if __name__ == "__main__":   
    f = open('../EngagementDetector/cert_new_openmpt.txt', 'rt')
    lines = f.readlines()
    f.close()

    #frameToLabelMap = getLabels()

    f = open('restricted_cert_values.txt', 'wt')
    for line in lines:
        if "UCSD" in line:
            continue
        cols = line.split('\t')
        filename = cols[0]
        subject = getSubject(filename)
        if os.path.exists("Spring2011/frames/{}".format(subject)):
            if filename in frameToLabelMap.keys():
                f.write(str(frameToLabelMap[filename]) + "\t" + line)
    f.close()
