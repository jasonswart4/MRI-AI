import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from functions_ import *
from skimage import io
from architectures_ import *

# Load pickled data
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

# Split data into images and labels
images = np.array(data['images']) # greyscale 256,256 images
labels = np.array(data['labels']) # greyscale where pixel values 0, 1, 2, 3 correspond to background, muscle, fat, bone

image_dim = images[0].shape[0] # image's dimension

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.05, random_state=420)

model = unet_model(input_shape=(image_dim, image_dim, 1))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

try:
    losses = []
    i_epoch = 0
    training_data = [[], [], []] # image, loss, epoch
    while True:
        subset_size = 10  # Adjust as needed
        num_epochs = 5  # Number of training epochs

        #model.optimizer.lr.assign(0.0001)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Generate a random subset of the data for this epoch
            _, _, [subset_data, subset_labels] = generate_random_subset(X_train, y_train, subset_size)

            # Train the model on the subset for this epoch
            history = model.fit(subset_data, subset_labels, epochs=5, validation_split=0.2)  # Adjust validation as needed
        
        i_epoch += num_epochs
        
        test_loss, test_acc = model.evaluate(X_val, y_val)
        losses.append(test_loss)
        model.save('MRI_Segmentation_Model.h5')

        # Plot the image to learning progress
        first_validation_image = X_val[0]
        predicted_mask = model.predict(np.expand_dims(first_validation_image, axis=0))
        predicted_mask = predicted_mask.squeeze()[:,:,1:4]
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

        #io.imsave(r'C:\Users\n10766316\Desktop\Python\mriFinal\progress_images\epoch' + str(i_epoch) + r'.jpg', predicted_mask)
        training_data[0].append(predicted_mask)
        training_data[1].append(test_loss)
        training_data[2].append(i_epoch)
        with open('training_data.pkl', 'wb') as file:
            pickle.dump(training_data, file)

        pass

except KeyboardInterrupt:
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"Test accuracy: {test_acc}")

    # Save the model for future use
    model.save('MRI_Segmentation_Model.h5')

    # Load the first validation image (assuming you have already split your data)
    first_validation_image = X_val[0]  # Assuming X_val contains your validation images

    # Use the trained model to make predictions
    predicted_mask = model.predict(np.expand_dims(first_validation_image, axis=0))

    # Plot the input and predicted output side by side
    plt.figure(figsize=(12, 6))

    # Input Image
    plt.subplot(1, 2, 1)
    plt.imshow(first_validation_image.squeeze(), cmap='gray')  # Assuming grayscale image
    plt.title("Input Image")
    plt.axis('off')

    # Predicted Output (Segmentation Mask)
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask.squeeze()[:,:,1:4])
    plt.title("Predicted Segmentation")
    plt.axis('off')

    plt.show()

# To continue training run this

"""
while True:
    subset_size = 10  # Adjust as needed
    num_epochs = 5  # Number of training epochs
    model.optimizer.lr.assign(0.0001)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Generate a random subset of the data for this epoch
        _, _, [subset_data, subset_labels] = generate_random_subset(X_train, y_train, subset_size)

        # Train the model on the subset for this epoch
        history = model.fit(subset_data, subset_labels, epochs=5, validation_split=0.2)  # Adjust validation as needed
    
    i_epoch += num_epochs
    
    test_loss, test_acc = model.evaluate(X_val, y_val)
    losses.append(test_loss)
    model.save('MRI_Segmentation_Model.h5')

    # Plot the image to learning progress
    first_validation_image = X_val[0]
    predicted_mask = model.predict(np.expand_dims(first_validation_image, axis=0))
    predicted_mask = predicted_mask.squeeze()[:,:,1:4]
    predicted_mask = (predicted_mask * 255).astype(np.uint8)

    #io.imsave(r'C:\Users\n10766316\Desktop\Python\mriFinal\progress_images\epoch' + str(i_epoch) + r'.jpg', predicted_mask)
    training_data[0].append(predicted_mask)
    training_data[1].append(test_loss)
    training_data[2].append(i_epoch)
    with open('training_data.pkl', 'wb') as file:
        pickle.dump(training_data, file)
"""