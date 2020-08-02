#! -*- coding: utf-8 -*-
import glob
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

from tensorflow.keras.datasets import fashion_mnist
import keras 

FileNames = ["img1.npy", "img2.npy", "img3.npy"]
ClassNames = ["うさぎ", "いぬ", "ねこ"]
hw = {"height":32, "width":32}        # リストではなく辞書型 中かっこで囲む

def PreProcess():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # 28 x 28の画像がgrayscaleで1chなので、28, 28, 1にreshapeする
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # 0-255の整数値を0〜1の小数に変換する
    # MNISTって必ずこの処理入るけれど、意味あるのかな
    x_train = x_train.astype('float32')
    #x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    #x_valid /= 255
    x_test /= 255

    # one-hot vector形式に変換する
    y_train = keras.utils.to_categorical(y_train, 10)
    #y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)



################################
######### モデルの構築 #########
################################
def BuildCNN(ipshape=(28, 28, 1), num_classes=3):
    model = Sequential()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


################################
############# 学習 #############
################################
def Learning(model):
    (x_train, y_train), (x_test, y_test) = PreProcess()

    history = model.fit(x_train, y_train)
    
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

Learning(BuildCNN())