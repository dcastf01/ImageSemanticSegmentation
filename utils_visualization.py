import matplotlib.pyplot as plt
import numpy as np
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    
    if np.amax(display_list[i])<=1:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i],scale=True))
    else:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i],scale=False),vmin=0,vmax=NUM_CLASSES-1)
    
    plt.axis('off')
  plt.show()
