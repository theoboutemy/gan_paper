import tensorflow as tf



class DataAugmentation:
    
    """
    This class defines the data augmentation that is applied to each batch when the model is trained.
    """
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
        
    def resize_img(self, img, size):
        """
        This function resizes an input image.  
              
        Args:
            img (tf.Tensor): The input image tensor to be resized.
            size (tuple): A tuple specifying the target size.

        Returns:
            tf.Tensor: The resized image tensor.
        """
        img = tf.image.resize(img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

       
        return img

    def random_crop(self, img1, img2):
         """
        This function lets you perform random cropping on an image. 
        To do this, the collected image is slightly enlarged, then a window of the initial size is randomly selected. 
        This is similar to a zoom.
        
        Args:
            img1 (tf.Tensor): The first input image to be cropped.
            img2 (tf.Tensor): The second input image to be cropped.

        Returns:
            tuple: a tuple containing the cropped versions of `img1` and `img2`.
        """
        # img1 and img2 need to be TensorFlow tensors
        img1_shape = tf.shape(img1)
        img2_shape = tf.shape(img2)

        h, w = img1_shape[0], img1_shape[1]
        crop_h, crop_w = self.target_size

        # select a random part of the image
        top = tf.random.uniform([], 0, h - crop_h, dtype=tf.int32)
        left = tf.random.uniform([], 0, w - crop_w, dtype=tf.int32)
        img1_cropped = tf.image.crop_to_bounding_box(img1, top, left, crop_h, crop_w)
        img2_cropped = tf.image.crop_to_bounding_box(img2, top, left, crop_h, crop_w)

        return img1_cropped, img2_cropped
    
    
    def synchronized_data_augmentation(self, batch_image1, batch_image2):
         """
         This function is used to generate the data increase for a complete batch.
        The function is synchronised in the sense that the same operations must be applied to the image and the associated semantic map.
        
        Args:
            batch_image1 (tf.Tensor): A batch of images 
            batch_image2 (tf.Tensor): A batch of corresponding semantic maps

        Returns:
            tuple: Two tensors representing the augmented batches of images and semantic maps.
        """
        
        #initializing empty lists
        augmented_batch_image1 = []
        augmented_batch_image2 = []

        for img1, img2 in zip(batch_image1, batch_image2):
            # randomly select the transformations that are applied
            flip_prob_left_right = tf.random.uniform([], 0, 1) > 0.5
            flip_prob_up_down = tf.random.uniform([], 0, 1) > 0.5
            angle = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            
            
            img1 = self.resize_img(img1, (290, 290))#taget
            img2 = self.resize_img(img2, (290, 290))#input

            # apply the same transformations to both the image and the associated semantic map
            if flip_prob_left_right:
                img1 = tf.image.flip_left_right(img1)
                img2 = tf.image.flip_left_right(img2)

            if flip_prob_up_down:
                img1 = tf.image.flip_up_down(img1)
                img2 = tf.image.flip_up_down(img2)

            img1 = tf.image.rot90(img1, k=angle)
            img2 = tf.image.rot90(img2, k=angle)
            
            img1, img2 = self.random_crop(img1, img2)

            augmented_batch_image1.append(img1)
            augmented_batch_image2.append(img2)

        return tf.stack(augmented_batch_image1), tf.stack(augmented_batch_image2)
