from PIL import Image as PILImage
import pickle
from IPython.display import display, Image

#------------------------------------------------------------------------#
# CREATE GIF FROM NETWORK PREDICTION IMAGES OBTAINED THROUGH TRAINING
#------------------------------------------------------------------------#

# Load data
with open('training_data.pkl', 'rb') as file:
    training_data = pickle.load(file)

# Your list of NumPy arrays (assuming they represent RGB images)
image_list = training_data[0]  # Replace with your actual images

# Convert NumPy arrays to PIL Images
image_list = [PILImage.fromarray(img) for img in image_list]

# Set the path for the output GIF
output_path = "timelapse.gif"

# Save the images in the list as a GIF
image_list[0].save(output_path, save_all=True, append_images=image_list[1:], duration=20, loop=0)

# Specify the path to your GIF
gif_path = "timelapse.gif"

# Display the GIF in the Jupyter Notebook
display(Image(filename=gif_path))