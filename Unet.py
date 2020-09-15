from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Input, Activation, BatchNormalization
from keras.models import Model
import os
import random
import numpy as np
import tensorflow as tf

# set_seed
seed = 6
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


class Unet:
    def __init__(self, height, width, channels, resnet=False):
        self.height = height
        self.width = width
        self.channels = channels
        self.filters = [16, 32, 64, 128, 256]
        self.resnet = resnet

    def downsample_layer(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
        c = Activation('relu')(c)
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(c)
        if self.resnet:
            res = Conv2D(filters, kernel_size=(1, 1), padding=padding, use_bias=False)(x)
            c = Add()([res, c])
        c = Activation('relu')(c)
        p = MaxPool2D((2, 2), (2, 2))(c)

        return c, p

    def upsample_layer(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = UpSampling2D((2, 2))(x)
        concat = Concatenate()([us, skip])
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(concat)
        c = Activation('relu')(c)
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(c)
        if self.resnet:
            res = Conv2D(filters, kernel_size=(1, 1), padding=padding, use_bias=False)(concat)
            c = Add()([res, c])
        c = Activation('relu')(c)
        return c

    def middle_layer(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
        c = Activation('relu')(c)
        c = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(c)
        if self.resnet:
            res = Conv2D(filters, kernel_size=(1, 1), padding=padding, use_bias=False)(x)
            c = Add()([res, c])
        c = Activation('relu')(c)
        return c

    def get_model(self):
        inputs = Input((self.height, self.width, self.channels))
        p0 = inputs

        # Encoder
        encoder = []
        for index, filter in enumerate(self.filters[:-1]):
            if index == 0:
                encoder.append(self.downsample_layer(p0, filters=filter))
            else:
                encoder.append(self.downsample_layer(encoder[index - 1][1], filter))

        # Middle layer
        middle = self.middle_layer(encoder[-1][1], self.filters[-1])

        # Decoder
        decoder = []
        for index, filter in enumerate(list(reversed(self.filters))[1:]):
            if index == 0:
                decoder.append(self.upsample_layer(middle, encoder[-(index + 1)][0], filter))
            else:
                decoder.append(self.upsample_layer(decoder[index - 1], encoder[-(index + 1)][0], filter))

        # Output layer
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(decoder[-1])

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
