import os  
# import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image  
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

#原始图片的存储位置
orig_picture = './data/train/'
 

np.random.seed(10)
#需要的识别类型
classes = ['NonDemented','MildDemented','ModerateDemented','VeryMildDemented'] 
img_final=[]
img_lable=[]

for index, name in enumerate(classes): 
    class_path = orig_picture +"/"+ name+"/"  
    print(index,name)
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
        img = cv2.imread(img_path)
        image = cv2.resize(img, (180, 180))
        # plt.imshow(image)
        # plt.show()
        image = image/255.0
        image = img_to_array(image)
        img_final.append(image)
        # print(index)
        img_lable.append(index)
img_final = np.array(img_final)
img_lable = np.array(img_lable)
print(img_final.shape,img_lable.shape)
x_train = img_final
y_train = img_lable
# x_train = img_final[:1000]
# y_train = img_lable[:1000]
# print(y_train)
y_train = to_categorical(y_train)
def baseconvolu(inputs):
    x=Conv2D(32, kernel_size=(3, 3), padding="same",input_shape=x_train.shape[1:], activation="relu")(inputs)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(64, kernel_size=(3, 3), padding="same",activation="relu")(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(512, activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(4, activation="softmax")(x)
    return x
def residual_block(x, filters, pooling=False):
    residual = x
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if pooling:
        x = MaxPooling2D(2, padding="same")(x)
        residual = Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = Conv2D(filters, 1)(residual)
    x = add([x, residual])
    return x
def resnet(inputs):
    x = Conv2D(64, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = residual_block(x, filters=128, pooling=True)
    x = residual_block(x, filters=256, pooling=True)
    x = residual_block(x, filters=512, pooling=True) 
    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation="softmax")(x)
    return outputs
# def resnet50_model(inputs):
# 	#include_top为是否包括原始Resnet50模型的全连接层，如果不需要自己定义可以设为True
# 	#不需要预训练模型可以将weights设为None
#     resnet50=tf.keras.applications.ResNet50(include_top=False,
#                                             weights=None,
#                                             input_shape=(180,180,3),
#                                             )
# 	#设置预训练模型冻结的层，可根据自己的需要自行设置                                                                                      
#     for layer in resnet50.layers[:15]:
#         layer.trainable = False  #

# 	#选择模型连接到全连接层的位置
#     last=resnet50.get_layer(index=10).output
#     #建立新的全连接层
#     x=Flatten(name='flatten')(last)
#     x=Dense(1024,activation='relu')(x)
#     x=Dropout(0.5)(x)
#     x=Dense(128,activation='relu',name='dense1')(x)
#     x=Dropout(0.5,name='dense_dropout')(x)
#     x=Dense(4,activation='softmax')(x)
#     return x
#     # model = tf.keras.models.Model(inputs=resnet50.input, outputs=x)
#     # model.summary() #打印模型结构
#     return model
inputs = keras.Input(shape=(180, 180, 3))
output = resnet(inputs)
model=keras.Model(inputs,output)
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 训练模型
history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=128, verbose=2)