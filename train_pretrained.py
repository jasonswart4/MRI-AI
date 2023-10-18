import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
from architectures_ import *
from loss_functions_ import *
from functions_ import *

architecture = resunet

# Load pickled data
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

# Split data into images and labels
images = data['images'] # greyscale 256,256 images
labels = np.array(data['labels']) # greyscale where pixel values 0, 1, 2, 3 correspond to background, muscle, fat, bone

# If the architecture needs an input shape of 3 reshape the images to RGB
if architecture().input_shape[3] == 3:
    images = [gray2rgb(images[i]) for i in range(len(images))]
images = np.array(images)

image_dim = images[0].shape[0] # image's dimension

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(images, labels, train_size=0.90, random_state=420)

model = architecture()

# Compile the model
model.compile(
    optimizer='adam',  # You can use any optimizer of your choice
    loss='sparse_categorical_crossentropy',  # Use this loss function for integer-encoded labels
    metrics=['accuracy']  # Use this metric for integer-encoded labels
)


losses = []
counter = 0
training_data = [[[],[]], [], []] # images, loss, epoch

test_loss, test_acc = model.evaluate(X_val, y_val)
losses.append(test_loss)

try:
    while True:
        subset_size = 5  # Adjust as needed
        num_epochs = 10  # Number of training epochs


        # Generate a random subset of the data for this epoch
        _, _, [subset_data, subset_labels] = generate_random_subset(X_train, y_train, subset_size)

        # Train the model on the subset for this epoch
        history = model.fit(subset_data, subset_labels, epochs=num_epochs)  # Adjust validation as needed
        
        test_loss, test_acc = model.evaluate(X_val, y_val)
        losses.append(test_loss)
        
        model.save('MRI_Segmentation_Model.h5')
        '''

        # Plot the image to learning progress
        
        val_inputs = [X_val[0], X_val[1]]

        predicted_masks = [model.predict(np.expand_dims(val_inputs[i], axis=0)) for i in range(len(val_inputs))]
        predicted_masks = [predicted_masks[i].squeeze()[:,:,1:4] for i in range(len(val_inputs))]
        predicted_masks = [(predicted_masks[i] * 255).astype(np.uint8) for i in range(len(val_inputs))]
        
        training_data[0][0].append(predicted_masks[0])
        training_data[0][1].append(predicted_masks[1])
        training_data[1].append(test_loss)
        training_data[2].append(counter)
        with open('training_data.pkl', 'wb') as file:
            pickle.dump(training_data, file)
        '''
        counter += 1
        
        pass

except KeyboardInterrupt:
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"Test accuracy: {test_acc}")

    # Save the model for future use
    model.save('MRI_Segmentation_Model.h5')

    val_inputs = [X_val[0], X_val[1]]

    predicted_masks = [model.predict(np.expand_dims(val_inputs[i], axis=0)) for i in range(len(val_inputs))]

    for i in range(len(val_inputs)):
        # Plot the input and predicted output side by side
        plt.figure(figsize=(12, 6))

        # Input Image
        plt.subplot(1, 2, 1)
        plt.imshow(val_inputs[i].squeeze(), cmap='gray')  # Assuming grayscale image
        plt.title("Input Image")
        plt.axis('off')

        # Predicted Output (Segmentation Mask)
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_masks[i].squeeze()[:,:,1:4])
        plt.title("Predicted Segmentation")
        plt.axis('off')

        plt.show()