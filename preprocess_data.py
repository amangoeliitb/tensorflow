import sys
import os
import scipy
import numpy as np
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage


BLUR_AMOUNT = 5
FINAL_SIZE = 80

TRAIN = 'train/'
VALIDATE = 'valid/'
PNG = '.png'
LABELS = 'labels.txt'

DATASET = VALIDATE

try:
    if sys.argv[1] == 'TRAIN':
        print "Preprocessing training data"
        DATASET = TRAIN
    elif sys.argv[1] == 'VALID':
        print "Preprocessing validation data"
    else:
        print "Invalid argument .. quitting"
        sys.exit()
except:
    DATASET = VALIDATE

images = os.listdir(DATASET)
images.remove(LABELS)
images = [int(image[:-4]) for image in images]
images.sort()
images = [str(image) for image in images]


def process(image):
    # apply gaussian filter to image to make text wider
    image = gaussian_filter(image, sigma=BLUR_AMOUNT)
    # invert black and white because most of the image is white
    image = 255 - image
    # resize image to make it smaller
    image = scipy.misc.imresize(arr=image, size=(FINAL_SIZE, FINAL_SIZE))
    # scale down the value of each pixel
    image = image / 255.0
    # flatten the image array to a list
    return [item for sublist in image for item in sublist]


preprocessed = []


for item in images:
    image = np.array(io.imread(DATASET + item + PNG))
    image = process(image)
    preprocessed.append(image)

np.save(DATASET[:-1] + '_preprocessed', preprocessed)
