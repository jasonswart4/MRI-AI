# TF imports
import tensorflow as tf
from tensorflow.python.framework import ops

# useful imports
import pickle
import pandas as pd

from functions import *
from architectures import *

# load data
data = pickle.load(open('data.pkl', 'rb'))

# The network input size is set according to the images from the pickled data.
# From the format images script, the image size is set and images resized. 
input_layers = data[0][0].shape[2]
im_size = data[0][0].shape[0]

# Choose the network architecture
net = unet2(im_size, input_layers)
print(net.summary())

net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss={'Muscle': iou_loss, 'Fat': iou_loss, 'Bone': iou_loss},  # Pixel-wise cross-entropy loss
              metrics=['accuracy'])

net.save('unet.h5')

# The code in the try can be copied and run independently after a model has
# been saved. 
try:
    hist = pd.DataFrame(columns=['loss', 'Bone_loss', 'Muscle_loss', 'Fat_loss'])
    #net = tf.keras.models.load_model("unet.h5", custom_objects={'iou_loss': iou_loss})
    net.optimizer.lr.assign(0.0001)
    while True:
        for _ in range(5):
            history = net.fit(image_generator(data, 5, True),
                                    steps_per_epoch = 10,
                                    epochs = 10)
            hist = pd.concat([hist, pd.DataFrame(history.history)], ignore_index = True)
            ops.reset_default_graph()
        net.save('unet.h5')
except KeyboardInterrupt:
    net.save('unet1.h5')
    hist.iloc[:, 1:4].plot()