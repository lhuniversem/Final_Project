import os  
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image  
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,DepthwiseConv2D,AveragePooling2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D,ReLU
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers.advanced_activations import ReLU
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
# The storage location of the original image
train_picture = './data/train'
test_picture='./data/val'
image_size=64
np.random.seed(10)
# Required identification type
classes = ['NonDemented','MildDemented','ModerateDemented','VeryMildDemented'] 
train_img_final=[]
train_img_lable=[]
test_img_final=[]
test_img_lable=[]
for index, name in enumerate(classes): 
    class_path = train_picture +"/"+ name+"/" 
    print(index,name) 
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
        img = cv2.imread(img_path)
        image = cv2.resize(img, (image_size, image_size))
        image = image/255.0
        image = img_to_array(image)
        train_img_final.append(image)
        train_img_lable.append(index)
for index, name in enumerate(classes): 
    class_path = test_picture +"/"+ name+"/"  
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
        img = cv2.imread(img_path)
        image = cv2.resize(img, (image_size, image_size))
        image = image/255.0
        image = img_to_array(image)
        test_img_final.append(image)
        test_img_lable.append(index)

train_img_final = np.array(train_img_final)
train_img_lable = np.array(train_img_lable)
test_img_final = np.array(test_img_final)
test_img_lable = np.array(test_img_lable)
print(train_img_final.shape,train_img_lable.shape)
train_img_final,train_img_lable=shuffle(train_img_final,train_img_lable,random_state=10)
test_img_final,test_img_lable=shuffle(test_img_final,test_img_lable,random_state=10)
x_train = train_img_final
y_train = train_img_lable
y_train = to_categorical(y_train)
x_test=test_img_final
y_test=test_img_lable
y_test= to_categorical(y_test)

def depth_point_conv2d(x,s=[1,1,2,1],channel=[64,128]):
    """
    s:the strides of the conv
    channel: the depth of pointwiseconvolutions
    """
    
    dw1 = DepthwiseConv2D((3,3),strides=s[0],padding='same')(x)
    bn1 = BatchNormalization()(dw1)
    relu1 = ReLU()(bn1)
    pw1 = Conv2D(channel[0],(1,1),strides=s[1],padding='same')(relu1)
    bn2 = BatchNormalization()(pw1)
    relu2 = ReLU()(bn2)
    dw2 = DepthwiseConv2D((3,3),strides=s[2],padding='same')(relu2)
    bn3 = BatchNormalization()(dw2)
    relu3 = ReLU()(bn3)
    pw2 = Conv2D(channel[1],(1,1),strides=s[3],padding='same')(relu3)
    bn4 = BatchNormalization()(pw2)
    relu4 = ReLU()(bn4)
    
    return relu4

def pointwise_conv(x,s=[1,1],channel=512):
    dw1 = DepthwiseConv2D((3,3),strides=s[0],padding='same')(x)
    bn1 = BatchNormalization()(dw1)
    relu1 = ReLU()(bn1)
    pw1 = Conv2D(channel,(1,1),strides=s[1],padding='same')(relu1)
    bn2 = BatchNormalization()(pw1)
    relu2 = ReLU()(bn2)
    
    return relu2


def mobilenet(inputs):
    x=Conv2D(32,(3,3),strides = 2,padding="same")(inputs)
    x= BatchNormalization()(x)
    x=ReLU()(x)
    x=depth_point_conv2d(x,s=[1,1,2,1],channel=[64,128])
    # x=depth_point_conv2d(x,s=[1,1,2,1],channel=[128,256])
    # x=depth_point_conv2d(x,s=[1,1,2,1],channel=[256,512])
    # x=repeat_conv(x)
    # x=repeat_conv(x)
    x=pointwise_conv(x)
    x=depth_point_conv2d(x,s=[1,1,2,1],channel=[128,256])
    x=pointwise_conv(x,channel=256)
    # x=AveragePooling2D((7,7))(x)
    x=Flatten()(x)


    x=Dense(512, activation="relu")(x)
    x=Dropout(0.3)(x)
    x=Dense(256, activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(4, activation="softmax")(x)
    return x

inputs = keras.Input(shape=(image_size, image_size, 3))
output = mobilenet(inputs)
model=keras.Model(inputs,output)
adam =optimizers.Adam(0.00001)
sgd = optimizers.SGD(lr=0.002)
# Compilation Model
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
# Training model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=2)
# Saving model

epochs=range(len(history.history["accuracy"]))
plt.figure()
plt.plot(epochs,history.history["accuracy"],"b",label="Training acc")
plt.plot(epochs,history.history["val_accuracy"],"r",label="Validation acc")
plt.title("Traing and Validation accuracy")
plt.legend()
plt.savefig("./mobilenetacc.jpg")
plt.show()


plt.figure()
plt.plot(epochs,history.history["loss"],"b",label="Training loss")
plt.plot(epochs,history.history["val_loss"],"r",label="Validation val_loss")
plt.title("Traing and Validation loss")
plt.legend()
plt.savefig("./mobileloss.jpg")
plt.show()


model.save("mobilenet.h5")
# 加载模型
# model= load_model("mobile.h5")
# # 模型测试
# score=model.evaluate(x_test,y_test,verbose=0)  
# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])