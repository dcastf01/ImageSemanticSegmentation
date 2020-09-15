import tensorflow as tf 
from ..utils_visualization import display

#pendiente de introducir correctamente
def create_mask(pred_mask):

  pred_mask = tf.argmax(pred_mask, axis=-1)
  
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
  
      unique, counts = np.unique(tf.math.round(create_mask(pred_mask)), return_counts=True)
      print("pred_mask",dict(zip(unique, counts)))
      unique, counts = np.unique(tf.math.round(mask[0]), return_counts=True)
      print("mask",dict(zip(unique, counts)))
      
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))]
    )


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(train_dataset)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))