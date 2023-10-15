import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('cnn_data.pkl', 'rb') as file:
    cnn_data = pickle.load(file)

with open('unet_data.pkl', 'rb') as file:
    unet_data = pickle.load(file)

plt.figure()
plt.plot(10*np.array(unet_data[2]), unet_data[1])
plt.plot(10*np.array(cnn_data[2]), cnn_data[1])
plt.xlabel('Epoch')
plt.ylabel('Categorical Cross Entropy Loss')
plt.legend(['U-Net', 'Simple CNN'])