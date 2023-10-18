import tensorflow as tf


def unet(input_shape=(256,256,1), name='unet'):
    inputs = tf.keras.layers.Input(input_shape)
    n = 16

    # Encoder
    conv1 = tf.keras.layers.Conv2D(n, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(n, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(2*n, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(2*n, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(4*n, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(4*n, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool3)
    concat3 = tf.keras.layers.Concatenate()([up3, conv3])
    conv4 = tf.keras.layers.Conv2D(4*n, 3, activation='relu', padding='same')(concat3)
    conv4 = tf.keras.layers.Conv2D(4*n, 3, activation='relu', padding='same')(conv4)

    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    concat2 = tf.keras.layers.Concatenate()([up2, conv2])
    conv5 = tf.keras.layers.Conv2D(2*n, 3, activation='relu', padding='same')(concat2)
    conv5 = tf.keras.layers.Conv2D(2*n, 3, activation='relu', padding='same')(conv5)

    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    concat1 = tf.keras.layers.Concatenate()([up1, conv1])
    conv6 = tf.keras.layers.Conv2D(n, 3, activation='relu', padding='same')(concat1)
    conv6 = tf.keras.layers.Conv2D(n, 3, activation='relu', padding='same')(conv6)

    # Output layer
    outputs = tf.keras.layers.Conv2D(4, 1, activation='softmax')(conv6)  # 4 classes: background, muscle, fat, bone
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def simple_cnn(input_shape=(256,256,1), name='simple_cnn'):
    model = tf.keras.models.Sequential(name=name)
    n = 56
    # Initial Conv layer and pooling
    model.add(tf.keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Second Conv layer and pooling
    model.add(tf.keras.layers.Conv2D(2*n, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Upsampling and third Conv layer
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(2*n, (3, 3), activation='relu', padding='same'))

    # Upsampling and fourth Conv layer
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same'))
    
    
    # Output layer to 4 classes with the same spatial dimensions as input
    model.add(tf.keras.layers.Conv2D(4, (1, 1), activation='softmax', padding='same'))

    return model

def unet_plusplus_block(x, filters, num_convs):
    c = x
    for _ in range(num_convs):
        c = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
    return c

def unet_plusplus(input_shape=(256,256,1), name='unet_plusplus'):
    inputs = tf.keras.layers.Input(input_shape)
    n = 23
    
    # Encoder
    c1 = unet_plusplus_block(inputs, n, 2)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = unet_plusplus_block(p1, 2*n, 2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = unet_plusplus_block(p2, 4*n, 2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    # Nested skip pathways
    u1 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    concat1 = tf.keras.layers.Concatenate()([c2, u1])
    n1 = unet_plusplus_block(concat1, 2*n, 1)
    
    u2 = tf.keras.layers.UpSampling2D((2, 2))(n1)
    concat2 = tf.keras.layers.Concatenate()([c1, u2])
    n2 = unet_plusplus_block(concat2, n, 1)
    
    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(n2)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def residual_block(x, filters):
    skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    
    # Match the depth of 'skip' and 'x' using a 1x1 conv on 'skip'
    if skip.shape[-1] != x.shape[-1]:
        skip = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(skip)
    
    x = tf.keras.layers.Add()([x, skip])  # Residual connection
    return x


def resunet(input_shape=(256,256,1), name='resunet'):
    inputs = tf.keras.layers.Input(input_shape)
    n = 21
    
    # Encoder with residual blocks
    c1 = residual_block(inputs, n)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 2*n)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 4*n)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Decoder with upsampling
    u1 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    concat1 = tf.keras.layers.Concatenate()([u1, c2])
    c4 = residual_block(concat1, 2*n)
    
    u2 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    concat2 = tf.keras.layers.Concatenate()([u2, c1])
    c5 = residual_block(concat2, n)
    
    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(c5)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


#----------------------------------------------------------------------------#
# Pretrained architectures
#----------------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate

def mobilenetv2(input_shape=(224, 224, 3), num_classes=4, name='crappy_mobilenetv2'):
    # Define the MobileNetV2 base model without the top classification layers
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, alpha=1.0)

    # Freeze the base model's weights
    base_model.trainable = False

    # Start from the base model output
    x = base_model.output

    # Add a 1x1 convolutional layer to get the desired number of channels
    x = Conv2D(num_classes, (1, 1))(x)

    # Gradually upsample and refine
    for _ in range(5):  # 2^5 = 32
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(num_classes, (3, 3), padding='same', activation='relu')(x)

    segmentation_output = x
    
    # Create the segmentation model
    model = tf.keras.Model(inputs=base_model.input, outputs=segmentation_output, name=name)

    return model

def unet_mobilenetv2(input_shape=(224, 224, 3), num_classes=4, name = 'mobilenetv2'):
    # Load the MobileNetV2 model and use its layers as the encoder
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model's weights
    base_model.trainable = False
    
    # Encoder layers (we'll use these feature maps for skip connections)
    skip_layers = [
        'block_1_expand_relu',   # 128x128
        'block_3_expand_relu',   # 64x64
        'block_6_expand_relu',   # 32x32
        'block_13_expand_relu',  # 16x16
    ]
    encoder_outputs = [base_model.get_layer(name).output for name in skip_layers]
    
    # Start building the U-Net model
    x = base_model.output  # This will be the bottom of the U (smallest spatial dimension)

    # Decoder with skip connections
    for i, skip_out in enumerate(reversed(encoder_outputs)):
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, skip_out])
        
        # Optional: Add more Conv2D layers here for refinement if needed
        x = Conv2D(128 // (2 ** i), (3, 3), padding='same', activation='relu')(x)
    
    # Last upsampling to match the original input size
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    # Output layer
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    # Create the U-Net model
    model = tf.keras.Model(inputs=base_model.input, outputs=x, name=name)

    return model

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate, Conv2D
import tensorflow as tf

def unet_resnet50(input_shape=(224, 224, 3), num_classes=4, name='resnet50'):
    # Load the ResNet50 model and use its layers as the encoder
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model's weights
    base_model.trainable = False
    
    # Encoder layers (we'll use these feature maps for skip connections)
    skip_layers = [
        'conv2_block3_out',  # 64x64
        'conv3_block4_out',  # 32x32
        'conv4_block6_out',  # 16x16
        'conv5_block3_out',  # 8x8
    ]
    encoder_outputs = [base_model.get_layer(name).output for name in skip_layers]
    
    # Start building the U-Net model
    x = base_model.output

    # Decoder with skip connections
    for i, skip_out in enumerate(reversed(encoder_outputs)):
        if i != 0:
            x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, skip_out])
        
        # Reduced number of filters
        x = Conv2D(56 // (2 ** i), (3, 3), padding='same', activation='relu')(x)
    
    # Last upsampling and convolutional layers with reduced filters
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)

    # Output layer
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    # Create the U-Net model
    model = tf.keras.Model(inputs=base_model.input, outputs=x, name=name)

    return model

#-----------------------------------------------------------------------#
# CUSTOM LOSS
#-----------------------------------------------------------------------#


def weighted_sparse_categorical_crossentropy(class_weights=[0.4538254687442443, 0.8873443591151765, 1.7468628781510478, 10.299004314141127]):
    # Convert class_weights to a tensor
    class_weights_tensor = tf.convert_to_tensor(list(class_weights), dtype=tf.float32)

    def loss(y_true, y_pred):
        # Get the class weights for each entry in the batch
        weights = tf.gather(class_weights_tensor, tf.cast(y_true, tf.int32))
        
        # Compute the standard sparse categorical crossentropy
        sce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Apply the weights
        weighted_sce = weights * sce
        
        return tf.reduce_mean(weighted_sce)

    return loss