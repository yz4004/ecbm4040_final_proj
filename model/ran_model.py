#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout

from tensorflow.keras.models import Model


# from layer_func import *

    

class ResidualAttentionNetwork():

    def __init__(self, input_shape, n_classes, activation, p=1, t=2, r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self.p = p
        self.t = t
        self.r = r
    
    
    
            
    def model_56(self):
        
        '''
        create a keras model 
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)
        
        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3
        
        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=5,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)
            
        ## first attention module
        out = residual_unit(out,32,128,1)
        out = attention_module_step1(out,128,128,1)
        
        ## second attention module
        out = residual_unit(out,128,256,2)
        out = attention_module_step1(out,256,256,1)
        
        ## third attention module
        out = residual_unit(out,256,512,2)
        out = attention_module_step2(out,512,512,1)
        
        ## final output layers
        out = residual_unit(out,512,1024,1)
        out = residual_unit(out,1024,1024,1)
        out = residual_unit(out,1024,1024,1)
        
        # print(out.shape,"############")
        # assert out.shape[1,2] == [4,4]  True
        
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = AveragePooling2D(pool_size=(2, 2), strides=(1,1))(out)

        out = Flatten()(out)
        
        ## FC 
        out = Dense(1024,activation="relu")(out)
        out = Dropout(0.5)(out)
        out = Dense(10,activation="softmax")(out)


         
        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)
        
        return model
    
    
    
    
    def model_56_1(self):
        
        '''
        create a keras model 
        
        (p,t,r) (2,2,1)
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)
        
        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3
        
        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=3,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        # 32*32*32
        # out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)
            
        ## first attention module
        out = residual_unit(out,32,128,1)  # 32 * 32 * 128
        out = attention_module_step1(out,128,128,1)  # 32 32 128
        
        ## second attention module
        out = residual_unit(out,128,256,2)  # 16 16 256
        out = attention_module_step2(out,256,256,1) # 16 16 256
        
        ## third attention module
        out = residual_unit(out,256,512,2)  # 8 8 512
        out = attention_module_step3(out,512,512,1) # 8 8 512
        
        ## final output layers
        out = residual_unit(out,512,1024,1)
        out = residual_unit(out,1024,1024,1)
        out = residual_unit(out,1024,1024,1)
        
        # print(out.shape,"############")
        # assert out.shape[1,2] == [4,4]  True
        
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # out = Flatten()(out)
#         # FC 
#         out = Dense(1024,activation="relu")(out)
#         out = Dropout(0.5)(out)
        #out = Dense(10,activation="softmax")(out)

        out = AveragePooling2D(pool_size=(8,8), strides=(1,1))(out)
        out = Conv2D(10,kernel_size=1,strides=(1,1),padding="same" )(out)
        out = Flatten()(out)
        out = Activation('softmax')(out)


         
        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)
        
        return model
    
    
    
    def model_92(self):

        '''
        create a keras model 
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3

        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=3,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # 32*32*32
        # out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)

        ## first attention module
        out = residual_unit(out,32,128,1)  # 32 * 32 * 128
        out = attention_module_step1(out,128,128,1)  # 32 32 128


        ## second attention module
        out = residual_unit(out,128,256,2)  # 16 16 256
        out = attention_module_step2(out,256,256,1) # 16 16 256
        out = attention_module_step2(out,256,256,1)

        ## third attention module
        out = residual_unit(out,256,512,2)  # 8 8 512
        out = attention_module_step3(out,512,512,1) # 8 8 512
        out = attention_module_step3(out,512,512,1)
        out = attention_module_step3(out,512,512,1)

        ## final output layers
        out = residual_unit(out,512,1024,1)
        out = residual_unit(out,1024,1024,1)
        out = residual_unit(out,1024,1024,1)

        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        
        
        #out = AveragePooling2D(pool_size=(4,4), strides=(1,1))(out)
        #         out = Flatten()(out)
        #         # FC 
        #         out = Dense(1024,activation="relu")(out)
        #         out = Dropout(0.5)(out)
        #         out = Dense(10,activation="softmax")(out)

        out = AveragePooling2D(pool_size=(8,8), strides=(1,1))(out)
        out = Conv2D(10,kernel_size=1,strides=(1,1),padding="same" )(out)
        # print(out.shape)
        out = Flatten()(out)
        # print(out.shape)
        out = Activation('softmax')(out)

        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)

        return model
    
    
    def model_92_1(self):

        '''
        create a keras model 
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3

        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=3,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # 32*32*32
        # out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)

        ## first attention module
        out = residual_unit(out,32,128,1)  # 32 * 32 * 128
        out = attention_module_step1(out,128,128,1)  # 32 32 128


        ## second attention module
        out = residual_unit(out,128,256,2)  # 16 16 256
        out = attention_module_step2(out,256,256,1) # 16 16 256
        out = attention_module_step2(out,256,256,1)

        ## third attention module
        out = residual_unit(out,256,512,2)  # 8 8 512
        out = attention_module_step3(out,512,512,1) # 8 8 512
        out = attention_module_step3(out,512,512,1)
        out = attention_module_step3(out,512,512,1)

        ## final output layers
        out = residual_unit(out,512,1024,1)
        out = residual_unit(out,1024,1024,1)
        out = residual_unit(out,1024,1024,1)


        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        out = AveragePooling2D(pool_size=(4,4), strides=(1,1))(out)
        out = Flatten()(out)
        out = Dense(1024,activation="relu")(out)
        out = Dropout(0.5)(out)
        out = Dense(10,activation="softmax")(out)



        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)

        return model



        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)

        return model
    
    
    def model_92_2(self):

        '''
        create a keras model 
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3

        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=5,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # 32*32*32
        # out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)

        ## first attention module
        out = residual_unit(out,32,128,1)  # 32 * 32 * 128
        out = attention_module_step1(out,128,128,1)  # 32 32 128


        ## second attention module
        out = residual_unit(out,128,256,2)  # 16 16 256
        out = attention_module_step2(out,256,256,1) # 16 16 256
        out = attention_module_step2(out,256,256,1)

        ## third attention module
        out = residual_unit(out,256,512,2)  # 8 8 512
        out = attention_module_step3(out,512,512,1) # 8 8 512
        out = attention_module_step3(out,512,512,1)
        out = attention_module_step3(out,512,512,1)

        ## final output layers
        out = residual_unit(out,512,1024,1)
        out = residual_unit(out,1024,1024,1)
        out = residual_unit(out,1024,1024,1)


        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        
        
#         ## output plan1
#         out = AveragePooling2D(pool_size=(4,4), strides=(1,1))(out) # error
#         out = Flatten()(out)
#         out = Dense(1024,activation="relu")(out)
#         out = Dropout(0.5)(out)
#         out = Dense(10,activation="softmax")(out)


        ## output plan2
        out = AveragePooling2D(pool_size=(8,8), strides=(1,1))(out)
        out = Conv2D(10,kernel_size=1,strides=(1,1),padding="same" )(out)
        print(out.shape)
        out = Flatten()(out)
        print(out.shape)
        out = Activation('softmax')(out)



        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)

        return model
    
    
    
    def model_92_3(self):

        '''
        create a keras model 
        '''
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        residual_unit = self.residual_unit
        attention_module_step1 = self.attention_module_step1
        attention_module_step2 = self.attention_module_step2
        attention_module_step3 = self.attention_module_step3

        # Initial Layers before Attention Module
        out = Conv2D(32,kernel_size=3,strides=(1,1),padding="same",use_bias=False )(input_data)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # 32*32*32
        # out = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same" )(out)

        ## first attention module
        out = residual_unit(out,32,128,1)  # 32 * 32 * 128
        out = attention_module_step1(out,128,128,1)  # 32 32 128


        ## second attention module
        out = residual_unit(out,128,512,2)  # 16 16 512
        out = attention_module_step2(out,512,512,1) 
        out = attention_module_step2(out,512,512,1)

        ## third attention module
        out = residual_unit(out,512,2048,2)  # 8 8 2048
        out = attention_module_step3(out,2048,2048,1)
        out = attention_module_step3(out,2048,2048,1)
        out = attention_module_step3(out,2048,2048,1)

        ## final output layers
        out = residual_unit(out,2048,2048,1)
        out = residual_unit(out,2048,2048,1)
        out = residual_unit(out,2048,2048,1)


        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        
        
        
#         ## output plan1
#         out = AveragePooling2D(pool_size=(4,4), strides=(1,1))(out) # error
#         out = Flatten()(out)
#         out = Dense(1024,activation="relu")(out)
#         out = Dropout(0.5)(out)
#         out = Dense(10,activation="softmax")(out)


        ## output plan2
        # out 8 8 2048
        out = AveragePooling2D(pool_size=(8,8), strides=(1,1))(out)
        out = Conv2D(10,kernel_size=1,strides=(1,1),padding="same" )(out)
        out = Flatten()(out)
        out = Activation('softmax')(out)



        # Fully constructed model
        model = Model(inputs=input_data, outputs=out)

        return model
    
    
    
    def residual_unit(self, x, input_channels, output_channels, stride=1):

        """
        x: input (none, n, n, input_channels)

        output_channels = 4* input_channels

        residual unit structure is same as in table 2 in paper

        """

        # we assume input_channels equals to x.shape[3]
        # assert x.shape[-1] == input_channels
        # assert output_channels/4== input_channels
        
        plan = 2 # 1 2
        
        if(plan == 1 ):
            # plan 1
            # filter1 = int( output_channels/4 )
            # filter2 = int( output_channels/4 )
            # filter3 = output_channels
            filter1 = input_channels
            filter2 = input_channels
            filter3 = output_channels
        if(plan == 2):
            filter1 = input_channels
            filter2 = input_channels
            filter3 = output_channels
        
        
        
        # print(filter1,filter2,filter3)

        # layer1  (n,n,c/4) -> (n,n,c/4)  c is output_channels
        out = BatchNormalization()(x)
        out1 = Activation('relu')(out)
        out = Conv2D(filters=filter1, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out1)
        # print(out.shape)
        # print(x.shape[1],x.shape[2],filter1)
        # assert out.shape == (None,x.shape[1],x.shape[2],filter1)


        # layer2  (n,n,c/4) -> (n,n,c/4)   if stride=1
        #      (n,n,c/4) -> (n/2,n/2,c/4) if stride=2        
        # in keras.api, padding = 'same'  
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(filters=filter2, kernel_size=(3,3), padding='same',use_bias=False, strides=stride)(out)
        # assert out.shape == (None,x.shape[1]/stride,x.shape[2]/stride,filter2)

        # layer3 
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(filters=filter3, kernel_size=(1,1), padding='same',use_bias=False, strides=1)(out)
        # assert out.shape == (None,x.shape[1]/stride,x.shape[2]/stride,filter3)

        x_1 = BatchNormalization()(x)
        x_1 = Activation('relu')(x_1)

        ## since we may use stride = 2, then residual output will be diffierent from input x by shape, use a conv to reshape x
        if(int(x.shape[-1])!=out.shape[-1] or stride!=1):
            # out = Add()([out, Conv2D(filters=output_channels, kernel_size=(1,1),padding='same', strides=stride)(out1)])
            out = Add()([out, Conv2D(filters=output_channels, kernel_size=(1,1),padding='same', strides=stride)(x_1)])
        else:
            out = Add()([out,out1])


        return out
    
    
    
    

    def attention_module_step1(self, x, input_channels, output_channels, stride=1):
        


        # input_channels, output_channels, stride are used for residual_unit()
        
        
        stride = 1  # in attention module, we keep the shape of input, only refine image in residual units between attention module
        
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
    
    
    def attention_module_step2(self, x, input_channels, output_channels, stride=1):

        # input_channels, output_channels, stride are used for residual_unit()
        
        
        stride = 1  # in attention module, we keep the shape of input, only refine image in residual units between attention module
        
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
        # maxpooling
        out_mask = MaxPool2D(pool_size=3,strides=2,padding="same")(x_)


        # 2*r block
        for _ in range(2*self.r):
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)
            
        # interpolation layer
        out_mask = UpSampling2D(size=(2, 2))(out_mask)
        
        for _ in range(self.r):  # self.r = 1
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)  ## teng 无此处，而是用 一个conv1 block
            
        
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
    
    
    
    def attention_module_step3(self, x, input_channels, output_channels, stride=1):

        # input_channels, output_channels, stride are used for residual_unit()
        
        
        stride = 1  # in attention module, we keep the shape of input, only refine image in residual units between attention module
        
        
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


        # 2*r block
        for _ in range(2*self.r):
            out_mask = self.residual_unit(x_, input_channels, output_channels, stride=1)

        
        for _ in range(self.r):  # self.r = 1
            out_mask = self.residual_unit(out_mask, input_channels, output_channels, stride=1)  ## teng 无此处，而是用 一个conv1 block
            
        
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
    

    
    def attention_residual_learning(self, mask_input, trunk_input):
        # https://stackoverflow.com/a/53361303/9221241
        Mx = Lambda(lambda x: 1 + x)(mask_input) # 1 + mask
        return Multiply()([Mx, trunk_input]) # M(x) * T(x)
   
    
    

    












