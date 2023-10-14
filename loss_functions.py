def iou_percent_error_loss(y_true, y_pred):
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, 4])  # Assuming 4 classes
    
    # Convert ground truth to one-hot encoding
    y_true_one_hot = tf.one_hot(tf.cast(y_true_flat, tf.int32), 4)  # Assuming 4 classes
    
    # Calculate the intersection for each class
    intersection = tf.reduce_sum(y_true_one_hot * y_pred_flat, axis=0)
    
    # Calculate the true total pixels for each class for normalization
    true_total_pixels = tf.reduce_sum(y_true_one_hot, axis=0)
    
    # Calculate absolute percentage error for each class' intersection
    epsilon = 1e-7  # Adding epsilon to avoid division by zero
    abs_percentage_error = tf.abs((true_total_pixels - intersection) / (true_total_pixels + epsilon))
    
    # Sum over all classes
    sum_abs_percentage_error = tf.reduce_sum(abs_percentage_error)
    
    # Normalize the error to [0, 1] by dividing by the number of classes
    num_classes = 4  # Assuming 4 classes
    normalized_error = sum_abs_percentage_error / num_classes
    
    return normalized_error

def dice_loss(y_true, y_pred):
    smooth = 1.0  # To avoid division by zero
    
    # Convert y_true and y_pred to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), 4)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dynamic_iou_loss(y_true, y_pred):
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, 4])  # Assuming 4 classes
    
    # Convert ground truth to one-hot encoding
    y_true_one_hot = tf.one_hot(tf.cast(y_true_flat, tf.int32), 4)  # Assuming 4 classes
    
    # Calculate the number of pixels for each class in the ground truth
    total_pixels = tf.reduce_sum(y_true_one_hot, axis=0)
    
    # Calculate dynamic class weights as the inverse of the class frequencies
    # Adding epsilon to avoid division by zero
    epsilon = 1e-7
    class_weights = 1.0 / (total_pixels + epsilon)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_one_hot * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_one_hot, axis=0) + tf.reduce_sum(y_pred_flat, axis=0) - intersection
    
    # Compute weighted IoU
    iou = (class_weights * intersection) / (class_weights * union + tf.keras.backend.epsilon())
    
    # Compute weighted IoU loss
    iou_loss = 1 - tf.reduce_sum(iou)
    
    return iou_loss

def weighted_iou_loss(y_true, y_pred):
    # Define class weights
    class_weights = [.1, .1, .2, .6]  # Weights for [background, muscle, fat, bone]
    
    # Flatten the tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])  # Assuming 4 classes
    
    # Convert ground truth to one-hot encoding
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), 4)  # Assuming 4 classes
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) - intersection
    
    # Compute weighted IoU
    iou = (class_weights * intersection) / (class_weights * union + tf.keras.backend.epsilon())
    
    # Compute weighted IoU loss
    iou_loss = 1 - tf.reduce_sum(iou)
    
    return iou_loss

def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)  # Cast to int32 to ensure the data type matches for one-hot encoding
    y_true_onehot = tf.one_hot(y_true, depth=4, axis=-1)  # One-hot encode y_true to have shape (None, 256, 256, 4)
    
    # Ensure the last axis sums to 1, i.e., it is a valid probability distribution
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    
    # Clip to prevent NaN and Inf
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Compute the categorical cross-entropy loss
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred))
    
    return loss

def absolute_difference_loss(y_true, y_pred):
    class_weights = [.1, .1, .1, .5]
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=y_pred.shape[-1])

    loss = tf.reduce_sum(class_weights*tf.abs(y_true - y_pred))
    return loss

def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)  # Cast to int32 to ensure the data type matches for one-hot encoding
    y_true_onehot = tf.one_hot(y_true, depth=4, axis=-1)  # One-hot encode y_true to have shape (None, 256, 256, 4)

    # Calculate the intersection and union for each class
    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true_onehot + y_pred, axis=[1, 2]) - intersection

    # Calculate the IoU for each class, avoiding division by zero
    iou = tf.math.divide_no_nan(intersection, union)

    # Average over the classes
    avg_iou = tf.reduce_mean(iou)
    
    return 1 - avg_iou  # return the loss, which is 1 - IoU