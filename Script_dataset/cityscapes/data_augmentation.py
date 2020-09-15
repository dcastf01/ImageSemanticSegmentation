
import tensorflow as tf

def resize(inimg,tgimg,heigth,width):

    inimg=tf.image.resize(inimg,[heigth,width])
    tfimg=tf.image.resize(tgimg,[heigth,width])
    
    return inimg,tfimg

def normalize(inimg,tgimg):
    inimg=(inimg/127.5)-1
    tgimg=(tgimg/127.5)-1

    return inimg,tgimg

def random_jitter(inimg,tgimg):

    inimg,tgimg=resize(inimg,tgimg,286,286)

    stacked_image=tf.stack([inimg,tgimg], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2,IMG_HEIGHT,IMG_WIDTH,3])
    
    inimg,tgimg=cropped_image[0],cropped_image[1]
    
    if tf.random.uniform(()) > 0.5 :
    
        inimg=tf.image.flip_left_right(inimg)
        tgimg=tf.image.flip_left_right(tgimg)
    return inimg,tgimg

def load_image(filename,augment=True):
  inimg=tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH+'/'+filename)),tf.float32)[..., :3]
  
  tgimg=tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH+'/'+filename)),tf.float32)[..., :3]
  
  inimg,tgimg=resize(inimg,tgimg,IMG_HEIGHT,IMG_WIDTH)
  if augment:
    inimg,tgimg = random_jitter(inimg,tgimg)
  inimg,tgimg= normalize(inimg,tgimg)
  return inimg, tgimg   
     
def load_train_image(filename):
  return load_image(filename,True)
def load_test_image(filename):
  return load_image(filename,False)