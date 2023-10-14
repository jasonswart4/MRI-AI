import numpy as np
from skimage import io, transform
from skimage.color import gray2rgb, rgb2gray
import pickle
import os
from functions import centrality, ImageAugmenter
import copy

imSize = 256

# image label are made using ImageJ. The label is a 1 layer image with pixel values serving as a label.
# eg. background pixels would be a 0, a pixel of bone would be a 1, a pixel of muscle would be a 2, etc.
# split_label creates 4 masks for the individual classes, background, bone, muscle, fat. 
def split_label(label):
    return [1.0*label == i+1 for i in range(len(np.unique(label)) - 1)]

def resize_image(image, new_size):
    resized_image = transform.resize(image, new_size, mode='constant', order=0)

    return resized_image

folder = r'C:\Users\n10766316\Desktop\Python\mriFinal\images_raw'

# Uncomment this to load test data
# folder = r'C:\Users\n10766316\Desktop\Python\mri\testData' 
inputImgs = []
for file in os.listdir(folder):
     image = io.imread(f'{folder}\{file}')
     image = image / 255
     if len(image.shape)==2:
          image = gray2rgb(image)

     image = resize_image(image, (imSize,imSize))

     inputImgs += [image]

folder = r'C:\Users\n10766316\Desktop\Python\mriFinal\labels_raw'
Bone = []
Muscle = []
Fat = []
for file in os.listdir(folder):
     image = resize_image(io.imread(f'{folder}\{file}'), (imSize,imSize))
     [muscle, fat, bone] = split_label(image)
     Muscle +=[muscle]
     Fat += [fat]
     Bone += [bone]

data = [inputImgs, Muscle, Fat, Bone]

# Add augmentations to dataset
augmenter = ImageAugmenter(copy.copy(data[0]))

data[0].extend(augmenter.adjust_contrast(gamma=0.5))
for i in range(3): data[i+1].extend(data[i+1])
data[0].extend(augmenter.adjust_contrast(gamma=2))
for i in range(3): data[i+1].extend(data[i+1])

pickle.dump(data, open('data.pkl', 'wb'))