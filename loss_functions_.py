import tensorflow as tf

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    class_weights = tf.constant([1.7584140742717782, 3.73578735460334, 7.145826000459102, 42.22489487724375])
    # Calculate the normal sparse categorical cross-entropy loss
    sce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Create a tensor of shape (batch_size,) where each entry contains the class weight for the corresponding label
    class_weights_tensor = tf.gather(class_weights, tf.cast(tf.squeeze(y_true), tf.int32))
    
    # Multiply the SCE loss by the class weights
    weighted_loss = sce * class_weights_tensor
    
    return weighted_loss

def weighted_sparse_categorical_crossentropy2(y_true, y_pred):
    """
    Compute the weighted sparse categorical crossentropy loss.
    
    Args:
        y_true (Tensor): ground truth labels.
        y_pred (Tensor): predicted labels.
        
    Returns:
        loss (Tensor): weighted loss value.
    """

    class_weights = tf.constant([1.8, 3.7, 7.1, 42.2])  # Example class weights for 4 classes
    
    # Reshape if needed
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Standard sparse categorical crossentropy
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    standard_loss = sce(y_true, y_pred)
    
    # Compute weights
    weights = tf.reduce_sum(class_weights * tf.one_hot(tf.cast(y_true, tf.int32), depth=len(class_weights)), axis=-1)
    
    # Compute weighted loss
    weighted_loss = standard_loss * weights
    
    return tf.reduce_mean(weighted_loss)

def weighted_sparse_categorical_crossentropy3(y_true, y_pred):
    """
    Compute weighted sparse categorical cross-entropy loss.
    
    Parameters:
    - y_true: Ground truth labels, shape of [batch_size, height, width]
    - y_pred: Predictions, shape of [batch_size, height, width, num_classes]
    - class_weights: Class weights, 1-D tensor of length num_classes
    
    Returns:
    - loss: Weighted sparse categorical cross-entropy loss
    """
    class_weights = tf.constant([1.8, 3.7, 7.1, 42.2], dtype=tf.float32)
    # Reshape y_true to shape [batch_size, height * width]
    y_true_reshaped = tf.reshape(y_true, [-1])
    
    # Reshape y_pred to shape [batch_size * height * width, num_classes]
    y_pred_reshaped = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Compute sparse categorical cross-entropy loss
    loss = tf.math.log(tf.reduce_sum(tf.exp(y_pred_reshaped), axis=-1))
    loss -= tf.reduce_sum(y_pred_reshaped * tf.one_hot(tf.cast(y_true_reshaped, tf.int32), tf.shape(y_pred)[-1]), axis=-1)
    
    # Compute weighted loss
    weights = tf.gather(class_weights, tf.cast(y_true_reshaped, tf.int32))
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)
    
    return loss

def weighted_sparse_categorical_crossentropy4(y_true, y_pred):
    weights = tf.constant([1.8, 3.7, 7.1, 42.2]) 

    # Compute standard sparse categorical cross-entropy
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Flatten the ground truth and predictions to 1D
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Create a weight vector by indexing class weights using ground truth labels
    weight_vector = tf.gather(weights, tf.cast(y_true_flat, tf.int32))
    
    # Compute the weighted loss
    weighted_ce = ce * tf.reshape(weight_vector, tf.shape(ce))
    
    return tf.reduce_mean(weighted_ce)