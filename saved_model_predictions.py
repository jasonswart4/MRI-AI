import os
import tensorflow as tf
from skimage import io
import numpy as np
from copy import copy
from skimage.color import gray2rgb, rgb2gray
from formatImages import resize_image
import matplotlib.pyplot as plt
from architectures_ import weighted_sparse_categorical_crossentropy as custom_loss

# Specify the folder where neural network models are stored
folder_path = r'C:\Users\n10766316\Desktop\Python\MRI-AI\models\\'
image_path = r'C:\Users\n10766316\OneDrive - Queensland University of Technology\Documents\MRI-AI_data\MRIs\\'
image = io.imread(image_path + r'im077.jpg')
image = rgb2gray(image)

nets = []
predictions = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):
        RGB = False
        # Construct the full path to the HDF5 file
        file_path = os.path.join(folder_path, filename)
        
        # Load the model from the HDF5 file using TensorFlow
        model = tf.keras.models.load_model(file_path, custom_objects={'loss': custom_loss})

        # input image checks
        shape = model.input_shape[1]
        if model.input_shape[3] == 3:
            RGB = True

        input_image = copy(image)

        input_image = resize_image(input_image, (shape, shape))
        if RGB:
            input_image = gray2rgb(input_image)

        input_image = np.expand_dims(input_image, axis=0)
        
        predictions.append(model.predict(input_image))
        
        # Append the loaded model to the list
        nets.append(model)

for i in range(len(predictions)):
    plt.imshow(predictions[i].squeeze()[:,:,1:4])
    plt.show()
    print(nets[i].name)

'''
for i in range(len(predictions)):
    R = 255*(predictions[i].squeeze()[:,:,1]>0.5)
    G = 255*(predictions[i].squeeze()[:,:,2]>0.5)
    B = 255*(predictions[i].squeeze()[:,:,3]>0.5)

    RGB = np.dstack((R,G,B))
    io.imshow(RGB)
    io.show()
    print(nets[i].name)
'''