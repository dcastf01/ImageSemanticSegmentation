import tensorflow as tf

class create_dataset():
    def __init__(self,dataset_train,dataset_test,dataset_validation,Which_dataset,
                IMAGE_SIZE:tuple,PERCENT_INCREMENTED_IN_JITTER:float,PROBABLITY_THRESHOLD:float):
        
        self.train=dataset_train.map(self.load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test=dataset_test.map(self.load_test_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.validation=dataset_validation.map(self.load_test_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.heigth=IMAGE_SIZE[0]
        self.width=IMAGE_SIZE[1]
        self.PERCENT_INCREMENTED_IN_JITTER=PERCENT_INCREMENTED_IN_JITTER
        self.PROBABLITY_THRESHOLD=PROBABLITY_THRESHOLD
        self.Which_dataset=Which_dataset

    def resize(self,inimg,tgimg,heigth,width):

        inimg=tf.image.resize(inimg,[heigth,width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tfimg=tf.image.resize(tgimg,[heigth,width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return inimg,tfimg

    def normalize(self,inimg,tgimg):
        
        inimg=(inimg/255)
        if self.Which_dataset=='oxford_iiit_pet':
            tgimg -= 1  

        return inimg,tgimg

    def random_jitter(self,inimg,tgimg,heigth,width):

    #this function increase image size then randomly crop the image, with this always we have different image.
    #After that we apply different technique of augmentation

        incremented=self.PERCENT_INCREMENTED_IN_JITTER
        heigth_incremented=int(heigth*(1+incremented))
        width_incremented=int(width*(1+incremented))

        original_shape = tf.shape(inimg)[-3:-1]
        num_image_channels = tf.shape(inimg)[-1]

        inimg,tgimg=self.resize(inimg,tgimg,heigth_incremented,width_incremented)
        stacked_image = tf.concat([inimg, tf.cast(tgimg, dtype=inimg.dtype)], axis=-1)
        num_stacked_channels = tf.shape(stacked_image)[-1]

        stacked_image = tf.image.random_crop(stacked_image, size=[heigth,width,num_stacked_channels])
        
        if tf.random.uniform(()) > self.PROBABLITY_THRESHOLD :
        
            stacked_image = tf.image.flip_left_right(stacked_image)
        
        inimg = stacked_image[..., :num_image_channels]
            # Resizing back to expected dimensions:
        inimg = tf.image.resize(inimg, original_shape,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Random B/S changes:
        if tf.random.uniform(()) > self.PROBABLITY_THRESHOLD :
            inimg = tf.image.random_brightness(inimg, max_delta=0.15)
        if tf.random.uniform(()) > self.PROBABLITY_THRESHOLD :
            inimg = tf.image.random_saturation(inimg, lower=0.05, upper=0.75)

        # inimg=tf.clip_by_value(tf.cast(inimg,dtype=tf.float32), 0.0, 1.0) 

        tgimg = tf.cast(stacked_image[..., num_image_channels:], dtype=tgimg.dtype)
        tgimg = tf.image.resize(tgimg, original_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return inimg,tgimg

    def load_image(self,datapoint,augment):
    
        input_image=datapoint['image']
        target_mask=datapoint['segmentation_mask']
        heigth,width=self.heigth,self.width
        input_image,target_mask = self.resize(input_image,target_mask,heigth,width)
        
        if augment:
            input_image,target_mask=self.random_jitter(input_image,target_mask,heigth,width)
            
        input_image, target_mask = self.normalize(input_image, target_mask)

        return input_image, target_mask

    def load_train_image(self,datapoint):
        return self.load_image(datapoint,True)
    def load_test_image(self,datapoint):
        return self.load_image(datapoint,False)
