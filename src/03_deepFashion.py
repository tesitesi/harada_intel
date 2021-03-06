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
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils

import keras 
import random 
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import scipy 
import cv2 
import matplotlib.pyplot as plt 

# 空リスト作成
x_train=[]
y_train=[]
x_test=[]
y_test=[]
# ラベル作成
y_dic = {0:"Denim", 1:"Jackets_Vests", 2:"Pants", 3:"Shirts_Polos", 4:"Shorts", 
5:"Suiting", 6:"Sweaters", 7:"Sweatshirts_Hoodies", 8:"Tees_Tanks"}

def PocFunc(f):
    f = np.float32(f)
    #RGBに分解
    b = f[:,:,0]
    g = f[:,:,1]
    r = f[:,:,2]
    #二次元離散フーリエ変換
    B = scipy.fft(b)
    G = scipy.fft(g)
    R = scipy.fft(r)
    # 強度で正規化、逆フーリエ変換、実部を取り出す
    b = np.real(scipy.ifft(B/np.abs(B)))
    g = np.real(scipy.ifft(G/np.abs(G)))
    r = np.real(scipy.ifft(R/np.abs(R)))
    f_new = np.stack([b,g,r],axis=2)
    return f_new
    




def PreProcess(pc=False):
    # ファイル検索
    x_path_list = glob.glob('/Users/tesiyosi/dev/harada_intel/dataset/MEN/*/*/??_1_front.jpg')
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    x = []
    y = []

    for path in x_path_list:
        # 画像取得
        im = np.array(Image.open(path))
        if pc == True:
            im = PocFunc(im)
        # 辞書を検索して分類を番号化
        lab = [k for k, v in y_dic.items() if v == Path(path).parts[-3]][0]
        x.append(im)
        y.append(lab)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100, stratify=y)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape(x_train.shape[0],256,256,3)
    x_test = x_test.reshape(x_test.shape[0],256,256,3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 9)
    y_test = keras.utils.to_categorical(y_test, 9)
    return (x_train, y_train), (x_test, y_test)


################################
######### モデルの構築 #########
################################
def BuildCNN():
    img_width, img_height = 256, 256
    nb_filters1 = 32
    nb_filters2 = 64
    conv1_size = 3
    conv2_size = 2
    pool_size = 2
    classes_num = 9
    lr = 0.0004

    model = Sequential()
    model.add(Conv2D(nb_filters1, kernel_size=(conv1_size, conv1_size), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(nb_filters2, kernel_size=(conv2_size, conv2_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=lr),
                metrics=['accuracy'])
    return model


################################
############# 学習 #############
################################
def Learning():
    model = BuildCNN()
    (x_train, y_train), (x_test, y_test) = PreProcess(pc=False)
    history = model.fit(x_train, y_train,epochs=10)
    model.summary()
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

Learning()