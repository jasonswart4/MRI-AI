import tensorflow as tf


def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    n = 8

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
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model