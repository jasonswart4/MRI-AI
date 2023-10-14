import matplotlib.pyplot as plt
import numpy as np
from random import randint, random
from skimage import transform, exposure, util, io, color
from skimage.color import rgb2gray, gray2rgb
from tensorflow.keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy
import tensorflow as tf
import cv2
import numpy as np

def plot_images(data):
    for n in range(len(data[0])): 
        images = [[data[0][n]], [data[1][n]], [data[2][n]], [data[3][n]]]
        plotHeight = min(10, len(images[0]))
        plotWidth = len(images)
        plt.figure(figsize=(2*plotWidth,2*plotHeight))
        for i, img in enumerate(images):
            for j , im in enumerate(img):
                ax = plt.subplot(plotHeight,plotWidth,1 + i + j*plotWidth)
                ax.imshow(im)
                ax.set_axis_off()
            
def plot_generator(generator):
    image, masks = next(generator)
    plotHeight = min(10, image.shape[0])
    plotWidth = len(masks)+1
    plt.figure(figsize=(8,2*plotHeight))
    for i, img in enumerate([image, *masks]):
        for j , im in enumerate(img):
            ax = plt.subplot(plotHeight,plotWidth,1 + i + j*plotWidth)
            ax.imshow(im)
            ax.set_axis_off()

def rotate(image, angle):
    channels = np.transpose(image, (2,0,1))
    changed = np.array([transform.rotate(i, angle) for i in channels])
    return np.transpose(changed, (1,2,0))

def augment(images):
    if random() > 0.5:
        images = [np.flip(i) for i in images]
    if random() > 0.5:
        images = [np.fliplr(i) for i in images]
    rotation = (random()-0.5)*180
    images[0] = rotate(images[0], rotation)
    images[1:] = [transform.rotate(i, angle=rotation) for i in images[1:]]
    return images

class ImageAugmenter:
    def __init__(self, images):
        self.images = images

    def adjust_contrast(self, gamma):
        augmented_images = [exposure.adjust_gamma(image, gamma=gamma) for image in self.images]
        return augmented_images

    def random_noise(self, var=0.01):
        augmented_images = [util.random_noise(image, var=var) for image in self.images]
        return augmented_images

def image_generator(data, batchSize, augmentation = False):
    while True:
        X = []
        Bone = []
        Muscle = []
        Fat = []
        
        for _ in range(batchSize):
            # Select a random image
            img = randint(0,len(data[0])-1)
            images = [i[img] for i in data]
            if augmentation:
                x,b,m,f = augment(images)
            else:
                x,b,m,f = images
            
            X += [x]
            Bone += [b]
            Muscle += [m]
            Fat += [f]
            
        batch_x = np.array(X)
        batch_y_bone = np.array(Bone)
        batch_y_muscle = np.array(Muscle)
        batch_y_fat = np.array(Fat)
            
        yield (batch_x, [1.0*batch_y_bone, 1.0*batch_y_muscle, 1.0*batch_y_fat])

def myBinaryFocalCrossentropy(gamma, alpha):
    BFCELoss = BinaryFocalCrossentropy()
    def loss(pred, true):
        pos_count = tf.reduce_sum(true)
        neg_count = tf.reduce_sum(1 - true)
        total_count = pos_count + neg_count
        pos_ratio = pos_count / total_count
        neg_ratio = neg_count / total_count
        imbalance = pos_ratio / neg_ratio 
        Loss = BFCELoss(pred, true)
        return Loss + (Loss / imbalance * alpha)
    return loss

def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)  # Adding a small epsilon to avoid division by zero
    return 1.0 - iou

def weighted_iou_loss(y_true, y_pred):
    # Define class weights
    class_weights = [.01, .01, .01, 1]  # Weights for [background, muscle, fat, bone]
    
    # Flatten the tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])  # Assuming 4 classes
    
    # Convert ground truth to one-hot encoding
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), 4)  # Assuming 4 classes
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) - intersection
    
    # Compute weighted IoU
    iou = (class_weights * intersection) / (class_weights * union + tf.keras.backend.epsilon())
    
    # Compute weighted IoU loss
    iou_loss = 1 - tf.reduce_mean(iou)
    
    return iou_loss

def centrality(image, region_size=0, intensity=0.2):
    image = rgb2gray(image)
    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Create empty arrays for x, y, and z coordinates
    x_coords = []
    y_coords = []
    z_values = []

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            # Define the region around the current pixel
            x_min = max(x - region_size // 2, 0)
            x_max = min(x + region_size // 2, width - 1)
            y_min = max(y - region_size // 2, 0)
            y_max = min(y + region_size // 2, height - 1)

            # Extract the region
            region = image[y_min:y_max + 1, x_min:x_max + 1]

            # Calculate the average pixel intensity in the region
            avg_intensity = np.mean(region) > intensity

            # Calculate pixel centrality based on distance from the center
            centrality = np.exp(0.00006 * (-(x - center_x) ** 2 - (y - center_y) ** 2))

            # Multiply average intensity by centrality
            intensity_with_centrality = avg_intensity #* centrality

            # Append the coordinates and intensity with centrality to the arrays
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(intensity_with_centrality)

    # Create an output image as a numpy array
    output_image = np.zeros((height, width))

    # Assign z_values to the corresponding pixel locations
    for x, y, z in zip(x_coords, y_coords, z_values):
        output_image[y, x] = z

    return output_image

def resize_image(image, new_size):
    resized_image = transform.resize(image, new_size, mode='constant', order=0)

    return resized_image

def prepare_single_input_image(image):
    imSize = 256
    image = image / 255
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    image = resize_image(image, (imSize,imSize))

    # The centrality function adds a 4th layer to the image
    # which gives the network a measure of how central the pixels are.
    input_image = np.dstack((
        color.rgb2gray(image),
        centrality(image, region_size=1, intensity=0.2),
        centrality(image, region_size=1, intensity=0.5)
    ))

    return input_image

def single_input_image(image):
    imSize = 256
    image = image / 255
    
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    image = resize_image(image, (imSize,imSize))

    return rgb2gray(image)