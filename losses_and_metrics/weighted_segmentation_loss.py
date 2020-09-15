import tensorflow as tf 
from .prepare_data_for_loss_and_metrics import  prepare_data_for_segmentation_loss
from loss_sparse_categorical_crossentropy import SegmentationLoss


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
    def __init__(self, weights=class_weights_log, ignore_value=255, 
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
