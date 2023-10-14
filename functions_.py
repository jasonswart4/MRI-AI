import numpy as np
import tensorflow as tf
import cv2
import random
from skimage import io

def generate_random_subset(data, labels, subset_size):
    """
    Generates a random subset of data and labels.

    Args:
        data (numpy.ndarray): The input data.
        labels (numpy.ndarray): The corresponding labels.
        subset_size (int): The desired size of the random subset.

    Returns:
        numpy.ndarray: The random subset of data.
        numpy.ndarray: The corresponding random subset of labels.
    """
    # Check if the subset size is not larger than the dataset size
    if subset_size > len(data):
        raise ValueError("Subset size is larger than the dataset size.")

    # Generate random indices to select the subset
    random_indices = np.random.choice(len(data), subset_size, replace=False)

    # Select the subset of data and labels based on the random indices
    subset_data = data[random_indices]
    subset_labels = labels[random_indices]

    augmented_subset = [[],[]]
    for i in range(len(subset_data)):
        [aug_image, rotation_angle] = augment_image(subset_data[i])
        augmented_subset[0].append(aug_image)
        # rotate the label
        augmented_subset[1].append(rotate_image(subset_labels[i], rotation_angle))

    return subset_data, subset_labels, [np.array(augmented_subset[0]), np.array(augmented_subset[1])]

def rotate_image(image, angle_degrees):
    # Get image dimensions
    height, width = image.shape

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return rotated_image

def add_noise_to_image(image, intensity):
    """
    Add specks of noise to a normalized image.

    Parameters:
    - image: Normalized input image (values in range [0, 1]).
    - intensity: Intensity of the noise (0.0 to 1.0).

    Returns:
    - Noisy image (still in the range [0, 1]).
    """
    noisy_image = np.copy(image)
    height, width = image.shape[:2]
    
    # Calculate the number of noisy pixels to add based on intensity
    num_noisy_pixels = int(intensity * height * width)
    
    # Generate random pixel coordinates for noisy pixels
    noisy_coords = np.random.randint(0, height, num_noisy_pixels), np.random.randint(0, width, num_noisy_pixels)
    
    # Set those pixels to random noise values in the range [0, 1]
    noisy_image[noisy_coords] = np.random.rand(num_noisy_pixels)
    
    return noisy_image

def scale_intensity(image, scale_factor):
    """
    Scale the intensity (brightness) of an image by a specified factor.

    Parameters:
    - image: Input image.
    - scale_factor: Factor by which to scale the intensity.

    Returns:
    - Scaled image.
    """
    scaled_image = np.copy(image)

    # Scale the intensity by multiplying all pixel values by the scale factor
    scaled_image *= scale_factor

    # Ensure pixel values are still in the valid range [0, 255]
    scaled_image = np.clip(scaled_image, 0, 1)

    # Convert back to the appropriate data type (e.g., uint8)
    #scaled_image = scaled_image.astype(np.uint8)

    return scaled_image

def augment_image(image):
    rotation_angle = random.uniform(-180, 180)
    noise_intensity = random.uniform(0,0.1)
    brightness_scale = random.uniform(0.5,2)

    augmented = add_noise_to_image(image,noise_intensity)
    augmented = rotate_image(augmented, rotation_angle)
    #augmented = scale_intensity(augmented, brightness_scale)

    return [augmented, rotation_angle]