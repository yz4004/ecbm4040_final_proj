#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout

from tensorflow.keras.models import Model



# Residual Unit (Pre-Activation)
def residual_unit(x, input_channels, output_channels, stride=1):
    
    
    """
    x: input (none, n, n, input_channels)
    
    output_channels = 4* input_channels
    
    residual unit structure is same as in table 2 in paper
    
    """
    
    # we assume input_channels equals to x.shape[3]
    assert x.shape[-1] == input_channels
    assert output_channels/4== input_channels
    
    filter1 = output_channels/4
    filter2 = output_channels/4
    filter3 = output_channels
    
    # layer1  (n,n,c/4) -> (n,n,c/4)  c is output_channels
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv2D(filters=filter1, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out)
    assert out.shape == (None,x.shape[1],x.shape[2],filter1)
    
    
    # layer2  (n,n,c/4) -> (n,n,c/4)   if stride=1
    #      (n,n,c/4) -> (n/2,n/2,c/4) if stride=2        
    # in keras.api, padding = 'same'  
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=filter2, kernel_size=(3,3), padding='same',use_bias=False, strides=stride)(out)
    assert out.shape == (None,x.shape[1]/stride,x.shape[2]/stride,filter2)
    
    # layer3 
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=filter3, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out)
    assert out.shape == (None,x.shape[1]/stride,x.shape[2]/stride,filter3)

      

    
    ## since we may use stride = 2, then residual output will be diffierent from input x by shape, use a conv to reshape x
    if(int(x.shape[-1])!=out.shape[-1] or stride!=1):
        out = Add()([out, Conv2D(filters=output_channels, kernel_size=(1,1),padding='same', strides=stride)(x)])
    else:
        out = Add()([out,x])


    return out



class AttentionModule_stage1(tf.keras.Model):
    # input 16 16
    def __init__(self, input_channels, output_channels, stride):
        super(ResidualBlock, self).__init__()
        """
        x: input (none, n, n, input_channels)

        output_channels = 4* input_channels

        residual unit structure is same as in table 2 in paper

        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.filter1 = output_channels/4
        self.filter2 = output_channels/4
        self.filter3 = output_channels
        
        self.stride = 1
        
        

    def call(self, x):
        
        residual = x
        
        # out1 is for holding batch normalized x, for use in conv4 if out and residual have different shape  
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        # since we may use stride = 2, then residual output will be diffierent from input x by shape, use a conv to reshape x
        if(int(x.shape[-1])!=out.shape[-1] or self.stride!=1):
            
            residual = self.conv4(out1)
            # out = Add()([out, Conv2D(filters=output_channels, kernel_size=(1,1),padding='same', strides=stride)(x)])

        out += residual
        
        
        


        # input_channels, output_channels, stride are used for residual_unit()

        residual_unit = self.residual_unit
        attention_residual_learning = self.attention_residual_learning
        
        # Send input_x through 'p' residual_units, in our setting, p = 1
        for _ in range(self.p):
            x_ = self.residual_unit(x, input_channels, output_channels, stride=1)

        ##### Perform Trunk Branch Operation, in our setting, t = 2
        ##############################################################################################
        for _ in range(self.t):
            out_trunk = self.residual_unit(x_, input_channels, output_channels, stride=1)
            
                                                                 

                

        ###### Perform Mask Branch Operation
        ##############################################################################################
        # first maxpooling
        out_mask = MaxPool2D(pool_size=3,strides=2,padding="same")(x_)
        out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)  ## ++
        
        # self.r = 1
        for _ in range(self.r):
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)
           
        # second maxpooling
        out_mask = MaxPool2D(pool_size=3,strides=2,padding="same")(out_mask)
        
        # 2*r block
        for _ in range(2*self.r):
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)
            
        # interpolation layer
        out_mask = UpSampling2D(size=(2, 2))(out_mask)
        
        # self.r = 1
        for _ in range(self.r):
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)  ## teng 无此处，而是用 一个conv1 block
            
            
        # interpolation layer  
        out_mask = UpSampling2D(size=(2, 2))(out_mask)
        out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)
        
        # finally, followed by 2 conv layers, 
        # out_mask = BatchNormalization()(out_mask) ##?
        # out_mask = Activation('relu')(out_mask)
        
        out_mask = BatchNormalization()(out_mask)
        out_mask = Activation('relu')(out_mask)
        out_mask = Conv2D(filters=output_channels, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out_mask)
        
        out_mask = BatchNormalization()(out_mask)
        out_mask = Activation('relu')(out_mask)
        out_mask = Conv2D(filters=output_channels, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out_mask)
        out_mask = Activation('sigmoid')(out_mask)
        
        
        # mask + trunk : (1+M(x))T(x)
        ##############################################################################################
        out = self.attention_residual_learning(out_mask, out_trunk)
          
        for _ in range(self.p):
            out = self.residual_unit(out, input_channels, output_channels, stride=1)

        
        return out








    
    
    
    
    


    
    
    
    
    
    
    
    











