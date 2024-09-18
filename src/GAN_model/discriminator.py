#This code is inspired from https://www.tensorflow.org/tutorials/generative/pix2pix?hl=fr

import tensorflow as tf

class Discriminator(tf.keras.Model) : 
    
    """
    This class defines the discriminator structure, as well as its loss function
    """
    def __init__(self,input_size, num_filters=64, kernel_size=4, strides=2, alpha=0.2):
        super(Discriminator, self).__init__()


        self.input_size = input_size
        self.crossentropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.model = self.build_discriminator()

    
    def downsample(self,nb,filter_nb, size, batchnorm = True):
        """        
        This function defines a downsampling block

        Args:
            filter_nb (int): Number of filters for the convolutional layers.
            size (int): Size of the convolutional kernel.
            batchnorm (bool): Whether to include a BatchNormalization layer 

        Returns:
            tf.keras.Sequential: A Sequential model containing the downsampling block.
        """
        
        initializer = tf.random_normal_initializer(0. , 0.02)
        
        #initialize an empty Sequential Block with tensorflow
        result = tf.keras.Sequential(name=f'Downsampling_Block_{nb}')
        #add a convolution layer
        result.add(tf.keras.layers.Conv2D(filter_nb, size , strides =2, padding = 'same', 
                                          kernel_initializer = initializer, use_bias = False))
        if batchnorm:
            #if required, add a batch normalization layer
            result.add(tf.keras.layers.BatchNormalization())
        
        #add a leaky relu layer
        result.add(tf.keras.layers.LeakyReLU())
        return result
    
    
    
    def build_discriminator(self):
        """
        This method defines the discriminator structure.
        
        Returns:
            tf.keras.Model: The discriminator model
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        #define the input (ie the semantic map), and the target (ie the real image)
        input = tf.keras.layers.Input(shape= (self.input_size,self.input_size, 1), name = "Semantic_Map")
        target =tf.keras.layers.Input(shape = (self.input_size,self.input_size, 3), name = "Image")

        #concatenate both images
        concatenated_images = tf.keras.layers.concatenate([input,target], name = 'Concatenation_Layer') #size : batch size, 256, 256, 6
        
        #add 3 downsampling blocks
        down1 = self.downsample(1,64,4,False)(concatenated_images) #output size : batch size, 128, 128, 64
        down2 = self.downsample(2,128, 4)(down1)
        down3 = self.downsample(3,256,4)(down2)

        #add more blocks as described in the Pix2Pix paper
        zero_padding1 = tf.keras.layers.ZeroPadding2D(name = 'Zero_Padding_Layer')(down3)
        convolution = tf.keras.layers.Conv2D(512, 4, strides =1, kernel_initializer = initializer,
                                             use_bias = False, name = 'Convolution_Layer')(zero_padding1)
        batchnormalization = tf.keras.layers.BatchNormalization(name = 'Batch_Normalization_Layer')(convolution)
        leaky_relu = tf.keras.layers.LeakyReLU(name = 'Leaky_Relu_Layer')(batchnormalization)
        zero_padding2 = tf.keras.layers.ZeroPadding2D(name = 'Zero_Padding_Layer_2')(leaky_relu)
        output_layer = tf.keras.layers.Conv2D(1,4, strides = 1, 
                                        kernel_initializer = initializer, name = 'Output_Feature_Map')(zero_padding2)
        
        

        
        return tf.keras.Model(inputs = [input, target], outputs = output_layer)
    
    def discriminator_loss(self, real_images, generated_images):
       """
        This function defines the discriminator loss, which is the sum of two elements: the real one 
        calculated from the target and the fake one calculated from the generated output.
        
        Args:
            real_images (tf.Tensor): The  discriminator output from real images.
            generated_images (tf.Tensor): The discriminator output from generated images

        Returns:
            tf.Tensor: The total discriminator loss
        """

        
        real_loss = self.crossentropy_loss(tf.ones_like(real_images), real_images)
        fake_loss = self.crossentropy_loss(tf.zeros_like(generated_images), generated_images)
        total_loss = real_loss + fake_loss
        
        return total_loss
        
