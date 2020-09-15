import tensorflow as tf
from .prepare_data_for_loss_and_metrics import prepare_data_for_segmentation_loss

class SegmentationMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes,ignore_value=CITYSCAPES_IGNORE_VALUE,name='mIoU', **kwargs):
      super(SegmentationMeanIoU, self).__init__(num_classes=num_classes,name=name, **kwargs)
      self.num_classes=num_classes   
      self.ignore_value = ignore_value 

    def update_state(self, y_true, y_pred, sample_weight=None):
      
      y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=self.num_classes, 
                                                            ignore_value=self.ignore_value)
        # And since tf.metrics.mean_iou() needs the label maps, not the one-hot versions,
        # we adapt accordingly:
      y_pred = tf.argmax(y_pred, axis=-1)

      super().update_state(y_true, y_pred, sample_weight=None)
      
    def result(self):
      return super().result()

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      super().reset_states() 