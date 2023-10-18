import tensorflow as tf

def conv_block(input_tensor, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def upsample_concat_block(input_tensor, concat_tensor):
    upsampled = tf.keras.layers.UpSampling2D((2, 2))(input_tensor)
    return tf.keras.layers.concatenate([upsampled, concat_tensor])

def simplified_unet(input_size=(64, 64, 1)):
    inputs = tf.keras.layers.Input(input_size)

    # Contracting Path
    conv1 = conv_block(inputs, 16)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = conv_block(pool1, 32)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    bottleneck = conv_block(pool2, 64)

    # Expanding Path
    up_conv2 = upsample_concat_block(bottleneck, conv2)
    up_block2 = conv_block(up_conv2, 32)

    up_conv1 = upsample_concat_block(up_block2, conv1)
    up_block1 = conv_block(up_conv1, 16)

    # Output Layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(up_block1)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense

def simplified_mobilenetv2(input_shape=(224, 224, 3), num_classes=1000):
    input_tensor = Input(shape=input_shape)
    
    # Initial Convolution
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Depthwise Separable Convolution Blocks (Repeat for desired depth)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layer
    x = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    
    return model

model = simplified_mobilenetv2()

# Save the model to a .h5 file
model.save('simplified_mobilenetv2.h5')