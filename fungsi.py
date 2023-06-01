import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Activation, Dropout,LeakyReLU

def make_model():
    model = Sequential()
    kernel = 3
    size_w = 64
    size_h = size_w
    train_data = image_generator.flow_from_directory('/kaggle/input/punakawan/wayang/train',
                                                 target_size=(size_w, size_h),
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 color_mode='rgb')
    model.add(Conv2D(filters=64, kernel_size=kernel, input_shape=(size_w, size_h, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=kernel, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=kernel, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=kernel, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(train_data.num_classes, activation='softmax'))
    
    return model
