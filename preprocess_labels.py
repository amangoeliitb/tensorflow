import sys
import os
import scipy
import numpy as np
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage


TRAIN = 'train/'
VALIDATE = 'valid/'
LABELS = 'labels.txt'

DATASET = VALIDATE

try:
    if sys.argv[1] == 'TRAIN':
        print "Preprocessing labels for training data"
        DATASET = TRAIN
    elif sys.argv[1] == 'VALID':
        print "Preprocessing labels for validation data"
    else:
        print "Invalid argument .. quitting"
        sys.exit()
except:
    DATASET = VALIDATE

labels = []

with open(DATASET + LABELS) as f:
    for line in f:
        labels.append(int(line[:-1]))

distinct_labels = list(set(labels))
distinct_labels.sort()

preprocessed_labels = []

for label in labels:
    curr = [0] * len(distinct_labels)
    curr[label] = 1
    preprocessed_labels.append(curr)

np.save(DATASET[:-1] + '_preprocessed_labels', preprocessed_labels)
