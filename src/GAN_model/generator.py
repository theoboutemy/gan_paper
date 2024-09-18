#code inspired from https://www.tensorflow.org/tutorials/generative/pix2pix?hl=fr


import tensorflow as tf

class Generator : 
    """
    This class defines the Generator structure and loss.
    """
    def __init__(self):
        self.model = self.build_generator()
        self.crossentropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.Lambda = 150

    
    def upsample(self,nb,filter_nb, size, apply_dropout = False):
        """
        Defines an upsampling block, as described in the Pix2Pix paper
    
        Args:
            filter_nb (int): Number of filters 
            size (int or tuple): size of the convolutional kernel.
            apply_dropout (bool): Whether to apply dropout after batch normalization
    
        Returns:
            tf.keras.Sequential: A Sequential model representing the upsampling block.
        """
        initializer = tf.keras.initializers.RandomNormal(0. , 0.02)
        result = tf.keras.Sequential(name = f'Upsampling_Block_{nb}')
        result.add(tf.keras.layers.Conv2DTranspose(filter_nb, size, strides = 2, padding="same", 
                                                   kernel_initializer = initializer, use_bias = False))
        
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
            
        result.add(tf.keras.layers.ReLU())
        
        return result
    
    def downsample(self, nb,filter_nb, size, batchnorm=True):
       """
        Defines a downsampling block, as described in the Pix2Pix paper

        Args:
            filter_nb (int): Number of filters.
            size (int or tuple): Size of the convolutional kernel.
            batchnorm (bool): Whether to include batch normalization after convolution .

        Returns:
            tf.keras.Sequential: A Sequential model representing the downsampling block.
        """
     
        initializer = tf.keras.initializers.RandomNormal(0., 0.02)
    
        downsample_block = tf.keras.Sequential(name = f'Downsampling_Block_{nb}')
    
        downsample_block.add(tf.keras.layers.Conv2D(filter_nb, size, strides=2, padding='same',
                                          kernel_initializer=initializer))
        
        if batchnorm:
            downsample_block.add(tf.keras.layers.BatchNormalization())
        
        downsample_block.add(tf.keras.layers.LeakyReLU())
    
        return downsample_block

    
    def build_generator(self):
         """
        This function defines the structure of the generator.

        Returns:
            tf.keras.Model: The compiled Keras model representing the generator.
        """
        
        #defining the input,ie the semantic map
        inputs = tf.keras.layers.Input(shape= [256,256,1], name = "Semantic_Map")
        
        #apply 8 downsampling blocks to determine the main characteristics of the semantic map
        down_stack = [
        self.downsample(1,64, 4, batchnorm=False),
        self.downsample(2,128, 4),
        self.downsample(3,256, 4),
        self.downsample(4,512, 4),
        self.downsample(5,512, 4),
        self.downsample(6,512, 4),
        self.downsample(7,512, 4),
        self.downsample(8,512, 4)
        ]

        #define 7 upsampling blocks to build the output image in RGB
        up_stack = [
        self.upsample(1,512, 4, apply_dropout=True),
        self.upsample(2,512, 4, apply_dropout=True),
        self.upsample(3,512, 4, apply_dropout=True),
        self.upsample(4,512, 4),
        self.upsample(5,256, 4),
        self.upsample(6,128, 4),
        self.upsample(7,64, 4)    ]
    
        initializer = tf.keras.initializers.RandomNormal(0. , 0.02)
        #the output layer needs a tanh activation function so the generated image has pixel values in [-1;1]
        output_layer = tf.keras.layers.Conv2DTranspose(3, 4,
                                                 strides = 2, padding = "same",
                                                 kernel_initializer = initializer,
                                                 activation = 'tanh', name = 'Generated_Image')
        
        x= inputs
        skip_connections = []
        for down_block in down_stack:
            x= down_block(x)
            skip_connections.append(x)

        #define the skips connections of the UNet structure
        skip_connections = reversed(skip_connections[:-1])
        nb=1
        for up_block,skip_connection in zip(up_stack,skip_connections):
            nb+=1
            x = up_block(x)
            x = tf.keras.layers.Concatenate( name = f'Concatenate_Layer_{nb}')([x,skip_connection])       
        
        
        output_image_rgb = output_layer(x)
        return tf.keras.Model(inputs=inputs, outputs = output_image_rgb)
    
    
    

    def generator_loss(self,disc_generated_output, gen_output, target):
        """
        Defines the loss of the generator, which is the sum of the adversarial loss and the L1 loss

        Args:
            disc_generated_output (tf.Tensor): Discriminator's output for generated image.
            gen_output (tf.Tensor): Image generated by the generator.
            target (tf.Tensor): Ground truth image

        Returns:
            tuple: total_generator_loss, gan_loss, l1_loss.
        """
        gan_loss = self.crossentropy_loss(tf.ones_like(disc_generated_output) , disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs( target - gen_output))
        total_generator_loss = gan_loss + (self.Lambda * l1_loss)
        
        return total_generator_loss, gan_loss, l1_loss
  