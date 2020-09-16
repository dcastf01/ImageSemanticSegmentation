import datetime
import itertools
import io

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
def call_tensorboard(prefix_log=log_dir):
    log_dir = log_dir+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=1,
    embeddings_metadata=None)
    return tensorboard_callback


def callback_confusion_matrix(NUM_CLASSES,CLASSES_NAMES,VALIDATION_STEPS ,val_dataset,model,log_dir=log_dir):

    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def plot_to_image(figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        
        buf = io.BytesIO()
        
        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')
        
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        
        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
        
        return image

    def log_confusion_matrix(epoch,logs=None):
    # print("logs",logs)

    # def get_mask_for_valid_labels(y_true, num_classes, ignore_value=255):
    #     """
    #     Build a mask for the valid pixels, i.e. those not belonging to the ignored classes.
    #     :param y_true:        Ground-truth label map(s) (each value represents a class trainID)
    #     :param num_classes:   Total nomber of classes
    #     :param ignore_value:  trainID value of ignored classes (`None` if ignored none)
    #     :return:              Binary mask of same shape as `y_true`
    #     """
    #     mask_for_class_elements = y_true < num_classes
    #     mask_for_not_ignored = y_true != ignore_value
    #     mask = mask_for_class_elements & mask_for_not_ignored

    #     return mask
    
        def flatten_image(image):
            return tf.reshape(image,shape=image.shape[0]*image.shape[1])

    for batch_image,batch_mask in val_dataset.take(VALIDATION_STEPS):
        batch_pred_mask = model.predict(batch_image)
        # display([batch_image[0],batch_mask[0],batch_pred_mask[0]])
        for pred_mask,mask in zip(batch_pred_mask,batch_mask):
    
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            
            # unique, counts = np.unique(tf.math.round(pred_mask), return_counts=True)
            # print("pred_mask",dict(zip(unique, counts)))
            
            mask=tf.math.round(mask)
            unique, counts = np.unique(mask, return_counts=True)
            pred_mask=flatten_image(pred_mask)
            mask=flatten_image(mask)
            cm=tf.math.confusion_matrix(mask,pred_mask,NUM_CLASSES)
        
        
    
    figure = plot_confusion_matrix(cm.numpy(), class_names=CLASSES_NAMES)
    cm_image = plot_to_image(figure)
    
    
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    return mask,pred_mask

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

