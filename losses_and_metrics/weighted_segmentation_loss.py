import tensorflow as tf 
from .prepare_data_for_loss_and_metrics import  prepare_data_for_segmentation_loss
from .loss_sparse_categorical_crossentropy import SegmentationLoss



def calculate_class_weights_log(dataset,CLASSES_NAMES,STEPS_PER_EPOCH):
    num_classes=len(CLASSES_NAMES)
    all_pixels_per_class = tf.convert_to_tensor([0] * num_classes)

    for img, mask in dataset.take(STEPS_PER_EPOCH): # Iterating over full validation setÇ
        mask=tf.cast(mask,tf.int32)
        one_hot_labels = tf.cast(tf.one_hot(tf.squeeze(mask, -1), num_classes), tf.int32)
        num_pixels_per_class = tf.reduce_sum(one_hot_labels, axis=[0, 1, 2])
        all_pixels_per_class += num_pixels_per_class
            
    sum_pixels = tf.reduce_sum(all_pixels_per_class)

    class_proportions = all_pixels_per_class / sum_pixels
    class_weights = sum_pixels / all_pixels_per_class
    class_weights_log = tf.cast(tf.math.log(class_weights), tf.float32)
    log_begin_red, log_begin_green = '\033[91m', '\033[92m'
    log_begin_bold, log_end_format = '\033[1m', '\033[0m'

    print('{}{:13} {:5}  →   {:6}  {:13}{}'.format(
        log_begin_bold, 'class', 'propor', 'weight', 'log(weight)', log_end_format))
    for label, prop, weight, weight_log in zip(
        CLASSES_NAMES, 
        class_proportions.numpy(), class_weights.numpy(), class_weights_log.numpy()):
        print('{0:13} {4}{1:5.2f}%{6}  →  {5}{2:7.2f}{3:9.2f}{6}'.format(
            label.name, prop * 100, weight, weight_log,
            log_begin_red, log_begin_green, log_end_format))

    return class_weights_log

def prepare_class_weight_map(y_true, weights):
    """
    Prepare pixel weight map based on class weighing.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param weights:       1D tensor of shape (N,) containing the weight value for each of the N classes
    :return:              Weight map (e.g., of shape B x H x W)
    """
    y_true_one_hot = tf.one_hot(y_true, tf.shape(weights)[0])
    weight_map = tf.tensordot(y_true_one_hot, weights, axes=1)
    return weight_map


class WeightedSegmentationLoss(SegmentationLoss):
    def __init__(self, weights, ignore_value=255, 
                 from_logits=False, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'):
        super().__init__(ignore_value, from_logits, reduction, name)
        self.weights = weights
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._prepare_data(y_true, y_pred)
        
        y_weight = prepare_class_weight_map(y_true, self.weights)
        if sample_weight is not None:
            y_weight = y_weight * sample_weight
            
        loss = super().__call__(y_true, y_pred, y_weight)
        return loss
