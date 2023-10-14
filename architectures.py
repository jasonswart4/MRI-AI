# TF imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.python.framework import ops
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def autoencoder(imSize, input_layers):
    input_img = Input(shape = (imSize, imSize, input_layers))

    # encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    # compressed representation
    encoded = MaxPooling2D((2, 2), padding='same', name='bottleneck')(x)
    # at this point the representation is (4, 4, 4) i.e. 64-dimensional

    # decoder
    def decoder(net, name):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        # decoded = 
        return Conv2D(1, (3, 3), activation='sigmoid', padding = 'same', name = name)(x)

    muscle = decoder(encoded, 'Muscle')
    fat = decoder(encoded, 'Fat')    
    bone = decoder(encoded, 'Bone')

    return Model(inputs = input_img, outputs = [muscle, fat, bone], name='autoencoder')

def unet(imSize, input_layers):
    input_img = Input(shape=(imSize, imSize, input_layers))

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)

    # Bottleneck
    bottleneck = MaxPooling2D((2, 2))(conv3)

    # Decoder
    up1 = UpSampling2D((2, 2))(bottleneck)
    concat1 = concatenate([conv3, up1], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = concatenate([conv2, up2], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    up3 = UpSampling2D((2, 2))(conv5)
    concat3 = concatenate([conv1, up3], axis=-1)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat3)

    # Output layers
    muscle = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Muscle')(conv6)
    fat = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Fat')(conv6)
    bone = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Bone')(conv6)

    return Model(inputs=input_img, outputs=[muscle, fat, bone], name='unet')

def unet2(imSize, input_layers):
    input_img = Input(shape=(imSize, imSize, input_layers))

    max_filters = 32
    filter_size = (3,3)

    # Encoder
    conv1 = Conv2D(max_filters, filter_size, activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(max_filters/2, filter_size, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(max_filters/4, filter_size, activation='relu', padding='same')(pool2)

    # Bottleneck
    bottleneck = MaxPooling2D((2, 2))(conv3)

    # Decoder
    def decoder(bottleneck, name):
        up1 = UpSampling2D((2, 2))(bottleneck)
        concat1 = concatenate([conv3, up1], axis=-1)
        conv4 = Conv2D(max_filters, filter_size, activation='relu', padding='same')(concat1)
        up2 = UpSampling2D((2, 2))(conv4)
        concat2 = concatenate([conv2, up2], axis=-1)
        conv5 = Conv2D(max_filters/2, filter_size, activation='relu', padding='same')(concat2)
        up3 = UpSampling2D((2, 2))(conv5)
        concat3 = concatenate([conv1, up3], axis=-1)
        conv6 = Conv2D(max_filters/4, filter_size, activation='relu', padding='same')(concat3)
        return Conv2D(1, filter_size, activation='sigmoid', padding='same', name=name)(conv6)

    # Output layers
    muscle = decoder(bottleneck, 'Muscle')
    fat = decoder(bottleneck, 'Fat')
    bone = decoder(bottleneck, 'Bone')

    return Model(inputs=input_img, outputs=[muscle, fat, bone], name='unet2')