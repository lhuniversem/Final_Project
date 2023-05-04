import os  
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image  
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
#原始图片的存储位置
train_picture = './data/train/'
test_picture='./data/val'
np.random.seed(10)
#需要的识别类型
classes = ['NonDemented','MildDemented','ModerateDemented','VeryMildDemented'] 
test_img_final=[]
test_img_lable=[]
for index, name in enumerate(classes): 
    class_path = test_picture +"/"+ name+"/"  
    print(index,name) 
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
        img = cv2.imread(img_path)
        image = cv2.resize(img, (64, 64))

        image = image/255.0
        image = img_to_array(image)
        test_img_final.append(image)

        test_img_lable.append(index)
test_img_final = np.array(test_img_final)
test_img_lable = np.array(test_img_lable)
test_img_final,test_img_lable=shuffle(test_img_final,test_img_lable,random_state=10)
x_test=test_img_final
y_test=test_img_lable
y_test = to_categorical(y_test)

# 加载模型
modelm= load_model("mobilenet.h5")
modelr=load_model
# 模型测试
score=model.evaluate(x_test,y_test,verbose=0)  
predicted=model.predict(x_test)
predicted1=[]
for pred in predicted:
    a=[0,0,0,0]
    a[np.argmax(pred)]=1
    predicted1.append(a)
# print(predicted1,"yyyyyyyyyyyyyyyyy",y_test)
p = precision_score(y_test, predicted1, average='weighted')
r = recall_score(y_test, predicted1, average='weighted')
f1score = f1_score(y_test, predicted1, average='weighted')
print("precision:",p)
print("recall:",r)
print("f1score:",f1score)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])