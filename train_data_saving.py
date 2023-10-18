import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
import copy
from architectures_ import *
#from loss_functions_ import *
from functions_ import *
from formatImages import resize_image
from keras.metrics import sparse_categorical_accuracy


#nets = [unet, simple_cnn, unet_plusplus, resunet, mobilenetv2, unet_mobilenetv2, unet_resnet50 ]
nets = [unet, simple_cnn, unet_resnet50, unet_mobilenetv2]

custom_loss = weighted_sparse_categorical_crossentropy

for i_net in range(len(nets)):
    architecture = nets[i_net]

    # Load pickled data
    with open('data.pkl', 'rb') as file:
        data = pickle.load(file)

    # Split data into images and labels
    images = data['images'] # greyscale 256,256 images
    labels = data['labels'] # greyscale where pixel values 0, 1, 2, 3 correspond to background, muscle, fat, bone

    # If the architecture needs an input shape of 3 reshape the images to RGB
    if architecture().input_shape[1] != 256:
        shape = architecture().input_shape[1]
        images = [resize_image(images[i], (shape, shape)) for i in range(len(images))]
        labels = [resize_image(labels[i], (shape, shape)) for i in range(len(labels))]
        
    if architecture().input_shape[3] == 3:
        images = [gray2rgb(images[i]) for i in range(len(images))]


        
    images = np.array(images)
    labels = np.array(labels)

    image_dim = images[0].shape[0] # image's dimension

    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_train, X_val, y_train, y_val = train_test_split(images, labels, train_size=0.90, random_state=420)
    
    class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=y_train.flatten())
    class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3]}

    model = architecture()

    # Compile the model
    model.compile(
        optimizer='adam',  # You can use any optimizer of your choice
        loss=custom_loss(class_weight_dict.values()),  # Use this loss function for integer-encoded labels
        metrics=[sparse_categorical_accuracy,]  # Use this metric for integer-encoded labels
    )


    losses = []
    counter = 0
    training_data = [[], [], []] # loss, epoch, accuracy

    test_loss, test_acc = model.evaluate(X_val, y_val)
    losses.append(test_loss)
    try:
        while True:
            subset_size = 10  # Adjust as needed
            num_epochs = 5  # Number of training epochs


            # Generate a random subset of the data for this epoch
            _, _, [subset_data, subset_labels] = generate_random_subset(X_train, y_train, subset_size)

            # Train the model on the subset for this epoch
            history = model.fit(subset_data, subset_labels, validation_data=(X_val, y_val), epochs=num_epochs)  # Adjust validation as needed
            
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
            counter += num_epochs

            
            training_data[0].append(test_loss)
            training_data[1].append(counter)
            training_data[2].append(test_acc)
            path = r'C:\Users\n10766316\Desktop\Python\MRI-AI\training_data\\' + model.name + r'.pkl'
            with open(path, 'wb') as file:
                pickle.dump(training_data, file)

            
            model.save(r'C:\Users\n10766316\Desktop\Python\MRI-AI\models\\' 
                       + model.name + '.h5')
            '''
            n = 20
            if len(training_data[0]) > n:
                if np.mean(training_data[0][-20:]) < 0.11:
                    raise KeyboardInterrupt
            '''
            if counter > 3000:
                raise KeyboardInterrupt

            pass

    except KeyboardInterrupt:
        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print(f"Test accuracy: {test_acc}")

        # Save the model for future use
        model.save(r'C:\Users\n10766316\Desktop\Python\MRI-AI\models\\' 
                       + model.name + '.h5')

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


legend = []
for i in range(len(nets)):
    model = nets[i]()
    legend.append(model.name)
    
    path = r'C:\Users\n10766316\Desktop\Python\MRI-AI\training_data\\' + model.name + r'.pkl'
    with open(path, 'rb') as file:
        training_data = pickle.load(file)
    
    plt.figure('losses')
    plt.plot(training_data[1], training_data[0])

plt.legend(legend)
#plt.title('Loss over time of different models')
plt.xlabel('Epochs trained')
plt.ylabel('Categorical Cross Entropy Loss')
plt.xlim(0,3000)

legend = []
final_accuracy = []
for i in range(len(nets)):
    model = nets[i]()
    legend.append(model.name)
    
    path = r'C:\Users\n10766316\Desktop\Python\MRI-AI\training_data\\' + model.name + r'.pkl'
    with open(path, 'rb') as file:
        training_data = pickle.load(file)
    
    plt.figure('accuracies')
    plt.plot(training_data[1], 100*np.array(training_data[2]))
    
    n = 50
    x = [np.mean(100*np.array(training_data[2][i:i+n])) for i in range(len(training_data[1]) - n)]
    plt.figure('moving average')
    x_data = [i*5 for i in range(len(x))] # correctly scaled x_data
    plt.plot(x_data, x)
    
    final_accuracy.append(x[-1])

    plt.annotate(model.name, 
                 (3000, x[-1]), 
                 textcoords="offset points", 
                 xytext=(10,0), 
                 ha='left', 
                 va='center')

plt.figure('accuracies')
plt.legend(legend)
#plt.title('Accuracy over time of different models')
plt.xlabel('Epochs trained')
plt.ylabel('Accuracy (%)')
plt.xlim(0,3e3)

plt.figure('moving average')
plt.gca().patch.set_facecolor('#e7e7e7')
#plt.legend(legend)
#plt.title('Accuracy over time of different models')
plt.xlabel('Epochs trained')
plt.ylabel('Accuracy (%)')
plt.ylim(90.5,97)
plt.xlim(0, 4200)

# BAR GRAPH

# Data
# legend and final_accuracy give names and final accuracy as fraction

final_accuracy, legend = zip(*sorted(zip(final_accuracy, legend)))

# Create a bar graph
plt.figure('bar graph')
plt.gca().set_facecolor('#e7e7e7')
plt.bar(legend, final_accuracy, color='#6e79e4')

# Adding labels and a title
plt.xlabel('Network name')
plt.ylabel('Accuracy after training')
plt.xticks(rotation=-45)
#plt.ylim(min(final_accuracy)- 10, 100)

plt.show()

import pandas as pd
data = {
    'Network name': legend,
    'Accuracy after training': final_accuracy
}

df = pd.DataFrame(data)

print(df)