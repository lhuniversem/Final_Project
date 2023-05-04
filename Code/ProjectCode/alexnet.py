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
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
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


train_img_final = np.array(train_img_final)
train_img_lable = np.array(train_img_lable)

print(train_img_final.shape,train_img_lable.shape)
train_img_final,train_img_lable=shuffle(train_img_final,train_img_lable,random_state=10)

x_train = train_img_final
y_train = train_img_lable
y_train = to_categorical(y_train)

def alexnet(inputs):
    x=Conv2D(4, (11, 11), strides=(1, 1),padding='same', input_shape=(image_size, image_size, 3), activation='relu')(inputs)
    # x=MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(x)
    x=Conv2D(8, (5, 5), strides=(1, 1),  padding='same',activation='relu')(x)
    # # x=MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(x)
    x=Conv2D(8, (3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    # x=MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(x)
    x=Conv2D(16, (3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    # # x=MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(x)
    x=Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    # x=MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(x)
    x=Flatten()(x)
    x=Dense(512, activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(256, activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(4, activation="softmax")(x)
    return x

inputs = keras.Input(shape=(image_size, image_size, 3))
output = alexnet(inputs)
model=keras.Model(inputs,output)
model.summary()
adam =optimizers.Adam(0.0001)
# Compilation Model
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=2)

# Drawing
epochs=range(len(history.history["accuracy"]))
plt.figure()
plt.plot(epochs,history.history["accuracy"],"b",label="Training acc")
plt.plot(epochs,history.history["val_accuracy"],"r",label="Validation acc")
plt.title("Traing and Validation accuracy")
plt.legend()
plt.savefig("./alexnetacc.jpg")
plt.show()


plt.figure()
plt.plot(epochs,history.history["loss"],"b",label="Training loss")
plt.plot(epochs,history.history["val_loss"],"r",label="Validation val_loss")
plt.title("Traing and Validation loss")
plt.legend()
plt.savefig("./alexnetloss.jpg")
plt.show()

# Saving model
model.save("alexnet.h5")
# loading model
model= load_model("alexnet.h5")
