<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import RandomRotation, RandomZoom, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def make_model():
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    
    def data_augmentar():
        data_augmentation = Sequential([
            RandomRotation(factor=(-0.15, 0.15)),
            RandomZoom(height_factor=(-0.3, -0.1))
        ])

        assert(data_augmentation.layers[0].name.startswith('random_rotation'))
        assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
        assert(data_augmentation.layers[1].name.startswith('random_zoom'))
        assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))

        return data_augmentation

    def alzheimer_classifier(image_shape=IMG_SIZE, data_augmentation=data_augmentar()):
        image_shape = IMG_SIZE + (3,)
        base_model = EfficientNetB0(input_shape=image_shape,
                                    include_top=False, 
                                    weights='imagenet')

        base_model.trainable = True
        for layer in base_model.layers[:218]:
            layer.trainable = False

        inputs = tf.keras.Input(shape=image_shape)
        x = data_augmentation(inputs)
        x = Rescaling(scale=1./255)(x)  # Rescaling input values
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=5, activation="softmax")(x)

        model = Model(inputs, outputs)

        return model
    
    alzheimer_model = alzheimer_classifier(IMG_SIZE, data_augmentar())

    return alzheimer_model
=======
import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, os

def make_model_mobile():
    img_size = 224
    batch_size_train = 32
    batch_size_test = 1
    optimizer = Adam(learning_rate=0.00001)
    
    inp = Input(shape = (224,224,3))
    model_mobile = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    x1 = model_mobile(inp)
    x2 = GlobalAveragePooling2D()(x1)
    out = Dense(6, activation='softmax')(x2)
    
    model_mobile = Model(inputs = inp, outputs = out)
    model_mobile.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    model_mobile.fit(train_generator,
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     epochs=20,
                     validation_data = test_generator
    )
    
    return mobile_model
>>>>>>> 27fca911f37c2884c17a3ced190ec5f068f5268e
