from PIL import Image
import os

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        print('Error: Output folder doesn\'t exist')

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            try:
                # Open the image file
                with Image.open(os.path.join(input_folder, filename)) as img:
                    # Resize the image to 256x256 pixels
                    img = img.resize((256, 256))
                    # Convert the image to grayscale
                    img = img.convert('L')
                    # Save the processed image to the output folder
                    img.save(os.path.join(output_folder, filename))
                    print(f'{filename} processed successfully!')
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')

# image folder
main_extension = r'C:\Users\n10766316\OneDrive - Queensland University of Technology\Documents\MRI-AI_data'
input_folder = main_extension + r'\images_raw'
output_folder = main_extension + r'\images_processed'
process_images(input_folder, output_folder)

# label folder
input_folder = main_extension + r'\labels_raw'
output_folder = main_extension + r'\labels_processed'
process_images(input_folder, output_folder)

#--------------------------------------------------------------------------#
# AFTER PROCESSING THE IMAGES SO THAT THEY ARE GREYSCALE AND THE RIGHT SIZE
# USE PICKLE TO CREATE A DATASET THAT CAN BE USED BY THE NEURAL NET
#--------------------------------------------------------------------------#

import os
import numpy as np
import pickle
from skimage import io, color
import matplotlib.pyplot as plt

# Define the paths to your image and label folders
image_folder = 'C:\\Users\\n10766316\\Desktop\\Python\\mriFinal\\images_processed'
label_folder = 'C:\\Users\\n10766316\\Desktop\\Python\\mriFinal\\labels_processed'

# Initialize empty lists to store images and labels
images = []
labels = []

# Loop through the image files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        # Load and preprocess the image
        img_path = os.path.join(image_folder, filename)
        img = io.imread(img_path, as_gray=True)  # Load as grayscale
        img = img / 255.0  # Normalize the image
        
        images.append(img)

# Do the same for labels
for filename in os.listdir(label_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        # Load and preprocess the image
        label_path = os.path.join(label_folder, filename)
        lbl = io.imread(label_path)
        
        labels.append(lbl)

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Create a dictionary to store data
data = {'images': images, 'labels': labels}

# Save the data as a pickle file
pickle_file_path = r'C:\Users\n10766316\Desktop\Python\MRI-AI\data.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

print("Dataset created and saved as 'data.pickle'.")

# Loop through the data and plot images with corresponding labels
# can change to false if don't want
if True:
    # Define custom labels for classes
    class_labels = ['Muscle', 'Fat', 'Bone']

    for i in range(len(data['images'])):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Increase the width of the figure
        
        axes[0].imshow(data['images'][i], cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        label_plot = axes[1].imshow(data['labels'][i], cmap='viridis')
        axes[1].set_title('Label')
        axes[1].axis('off')
        
        # Create a colorbar (legend) separately for the label plot
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size of the colorbar
        legend = plt.colorbar(label_plot, cax=cax, ticks=[0, 1, 2])
        legend.set_ticks([1, 2, 3])  # Position the ticks in the middle of the color regions
        legend.set_ticklabels(class_labels)  # Assign labels to the ticks
        
        plt.show()
