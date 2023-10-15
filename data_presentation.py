from PIL import Image as PILImage
import pickle
from IPython.display import display, Image
import cv2
import copy
from formatImages import resize_image
from skimage import io
import matplotlib.pyplot as plt

def add_text_to_image(image, text="Sample Text"):
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Define font and size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_color = (255, 255, 255)  # White
    font_thickness = 3
    
    # Get the size the text will take up
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 20  # 20 pixels from the bottom

    # Add text to image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image

def add_text(image, text):
    image = copy.copy(image)
    image = resize_image(image, (1024, 1024))
    
    return add_text_to_image(image, text)

#------------------------------------------------------------------------#
# CREATE GIF FROM NETWORK PREDICTION IMAGES OBTAINED THROUGH TRAINING
#------------------------------------------------------------------------#

# Load data
with open('training_data.pkl', 'rb') as file:
    training_data = pickle.load(file)

# Your list of NumPy arrays (assuming they represent RGB images)
image_list = training_data[0][0]  # Replace with your actual images

for i in range(len(image_list)):
    image_list[i] = add_text(image_list[i], 'Epoch: ' + str(10*training_data[2][i]))

# Convert NumPy arrays to PIL Images
image_list = [PILImage.fromarray(img) for img in image_list]

# Set the path for the output GIF
output_path = "timelapse.gif"

# Save the images in the list as a GIF
image_list[0].save(output_path, save_all=True, append_images=image_list[1:], duration=200, loop=0)

# Specify the path to your GIF
gif_path = "timelapse.gif"

# Display the GIF in the Jupyter Notebook
display(Image(filename=gif_path))

io.imshow(training_data[0][0][-1])
plt.figure()
plt.plot(10*np.array(training_data[2]), training_data[1])
plt.title('Loss over time for simple CNN model')
plt.xlabel('Epoch')
plt.ylabel('Categorical Cross Entropy Loss')