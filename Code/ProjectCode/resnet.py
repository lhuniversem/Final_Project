import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image  
import numpy as np
import cv2
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # 按需分配显存
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 最大使用显存80%

from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
#原始图片的存储位置
train_picture = './data/train'
test_picture='./data/val'

imges_size=64
#需要的识别类型
classes = ['NonDemented','MildDemented','ModerateDemented','VeryMildDemented']
train_img_final=[]
train_img_label=[]
# test_img_final=[]
# test_img_label=[]
for index, name in enumerate(classes): 
    class_path = train_picture +"/"+ name+"/"  
    print(index,name)
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
        img = cv2.imread(img_path)
        image = cv2.resize(img, (imges_size, imges_size))
        # plt.imshow(image)
        # plt.show()
        image = image/255.0
        image = img_to_array(image)
        train_img_final.append(image)
        # print(index)
        train_img_label.append(index)
      
# for index, name in enumerate(classes): 
#     class_path = test_picture +"/"+ name+"/"  
#     for img_name in os.listdir(class_path):  
#         img_path = class_path + img_name  
#         img = cv2.imread(img_path)
#         image = cv2.resize(img, (imges_size, imges_size))
#         # plt.imshow(image)
#         # plt.show()
#         image = image/255.0
#         image = img_to_array(image)
#         test_img_final.append(image)
#         # print(index)
#         test_img_label.append(index)
np.random.seed(10) 
train_img_final = np.array(train_img_final)
train_img_label = np.array(train_img_label)
# test_img_final = np.array(test_img_final)
# test_img_label = np.array(test_img_label)
print(train_img_final.shape,train_img_label.shape)
train_img_final,train_img_label=shuffle(train_img_final,train_img_label,random_state=10)
# test_img_final,test_img_label=shuffle(test_img_final,test_img_label,random_state=10)
x_train = train_img_final
y_train = train_img_label
y_train = to_categorical(y_train)
# x_test=test_img_final
# y_test=test_img_label
# y_test= to_categorical(y_test)
def baseconvolu(inputs):
    x=Conv2D(32, kernel_size=(3, 3), padding="same",input_shape=x_train.shape[1:], activation="relu")(inputs)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(64, kernel_size=(3, 3), padding="same",activation="relu")(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(512, activation="relu")(x)
    x=Dense(256, activation="relu")(x)
    x=Dense(128, activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(4, activation="softmax")(x)
    return x
def residual_block(x, filters, pooling=False):
    residual = x
    x =Conv2D(filters, 3, padding="same",activation="relu")(x)
    # x=MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same",activation="relu")(x)
    # x=MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    if pooling:
        x = MaxPooling2D(2, padding="same")(x)
        residual = Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = Conv2D(filters, 1,activation="relu")(residual)
    x = add([x, residual])
    return x
def resnet(inputs):
    x = Conv2D(16, 3, padding="same",activation="relu")(inputs)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = Conv2D(64, 3, padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)   
    x = residual_block(x, filters=16, pooling=True)
    # x = residual_block(x, filters=256, pooling=True)
    # x = residual_block(x, filters=512, pooling=True) 
    x = Conv2D(16, 3, padding="same",activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(512, activation="relu")(x)
    x=Dense(256, activation="relu")(x)
    x=Dense(128, activation="relu")(x)
    x=Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation="softmax")(x)
    return outputs
inputs = keras.Input(shape=(imges_size, imges_size, 3))
output = resnet(inputs)
model=keras.Model(inputs,output)
adam =optimizers.Adam(0.0001)
sgd = optimizers.SGD(lr=0.01)
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
# 训练模型
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, batch_size=128, verbose=2)
#绘图
epochs=range(len(history.history["accuracy"]))
plt.figure()
plt.plot(epochs,history.history["accuracy"],"b",label="Training acc")
plt.plot(epochs,history.history["val_accuracy"],"r",label="Validation acc")
plt.title("Traing and Validation accuracy")
plt.legend()
plt.savefig("./acc.jpg")
plt.show()

plt.figure()
plt.plot(epochs,history.history["loss"],"b",label="Training loss")
plt.plot(epochs,history.history["val_loss"],"r",label="Validation val_loss")
plt.title("Traing and Validation loss")
plt.legend()
plt.savefig("./loss.jpg")
plt.show()
# 保存模型
model.save("resnet.h5")
# 加载模型
model= load_model("resnet.h5")
# 模型测试
# score=model.evaluate(x_test,y_test,verbose=0)  
# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])