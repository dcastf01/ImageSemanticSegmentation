import tensorflow as tf
from .prepare_data_for_loss_and_metrics import prepare_data_for_segmentation_loss

class SegmentationAccuracy(tf.keras.metrics.Accuracy):
    def __init__(self, name='acc',ignore_value=None, **kwargs):
      super(SegmentationAccuracy, self).__init__(name=name, **kwargs)
      self.ignore_value = ignore_value

    def update_state(self, y_true, y_pred, sample_weight=None):
      num_classes = y_pred.shape[-1]

      y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                          num_classes=num_classes, 
                                                          ignore_value=self.ignore_value)
      y_pred = tf.argmax(y_pred, axis=-1)
      
      super().update_state(y_true, y_pred, sample_weight=None)
      
    def result(self):
      return super().result()

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      super().reset_states()