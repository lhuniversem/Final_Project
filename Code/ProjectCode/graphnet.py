import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk,Image
import numpy as np
from keras.models import load_model
import cv2
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image  
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,add,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score


# The storage location of the original image
train_picture = './data/train/'
test_picture='./data/val'
np.random.seed(10)
# Required identification type
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

# loading model
modelr = load_model('resnet.h5')
modela = load_model('alexnet.h5')
modelm = load_model('mobilenet.h5')
# Model testing
predictedr=modelr.predict(x_test)
predicteda=modela.predict(x_test)
predictedm=modelm.predict(x_test)
predicted1=[]
predicted=zip(predictedr,predicteda,predictedm)
for predr,preda,predm in predicted:
    a=[0,0,0,0]
    a[np.argmax(predr)]=a[np.argmax(predr)]+1
    a[np.argmax(predm)]=a[np.argmax(predm)]+1
    a[np.argmax(preda)]=a[np.argmax(preda)]+1
    # print(a)
    index=np.argmax(a)
    a=[0,0,0,0]
    a[index]=1
    predicted1.append(a)
p = precision_score(y_test, predicted1, average='weighted')
r = recall_score(y_test, predicted1, average='weighted')
f1score = f1_score(y_test, predicted1, average='weighted')
acc=accuracy_score(y_test, predicted1)
print("precision:",p)
print("recall:",r)
print("f1score:",f1score)
print('accuracy:', acc)


classes ={
    0:'NonDemented',
    1:'MildDemented',
    2:'ModerateDemented',
    3:'VeryMildDemented'
}
def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image= im
    label.configure(text=' ')
    show_class_fy_button(file_path)

def show_class_fy_button(file_path):
    classify_btnr = Button(top,text="   Resnet  ",command=lambda:classifyr(file_path),padx=20,pady=5)
    classify_btnr.configure(background="#364156",foreground="white",font=('arial',10,'bold'))
    classify_btna = Button(top,text=" Alex net  ",command=lambda:classifya(file_path),padx=20,pady=5)
    classify_btna.configure(background="#364156",foreground="white",font=('arial',10,'bold'))
    classify_btnm = Button(top,text="Mobile net",command=lambda:classifym(file_path),padx=20,pady=5)
    classify_btnm.configure(background="#364156",foreground="white",font=('arial',10,'bold'))
    classify_btnv = Button(top,text="Vote",command=lambda:classifyv(file_path),padx=20,pady=5)
    classify_btnv.configure(background="#364156",foreground="white",font=('arial',10,'bold'))
    classify_btnr.place(relx=0.79,rely=0.4)
    classify_btna.place(relx=0.79,rely=0.5)
    classify_btnm.place(relx=0.79,rely=0.6)
    classify_btnv.place(relx=0.79,rely=0.7)
def classifyv(file_path):
    vote=[0,0,0,0]
    img = cv2.imread(file_path)
    image = cv2.resize(img, (64, 64))
    image = np.expand_dims(image,axis=0)
    image = np.array(image)
    pred1 = modelr.predict([image])[0]
    pred2 = modela.predict([image])[0]
    pred3 = modelm.predict([image])[0]
    pred1 = np.argmax(pred1)
    pred2 = np.argmax(pred2)
    pred3 = np.argmax(pred3)
    print(pred1,pred2,pred2)
    print(classes[pred1],classes[pred2],classes[pred3])
    vote[pred1]+=1
    vote[pred2]+=1
    vote[pred3]+=1
    pred=pred2
    pred0=max(vote)
    if pred0!=1:
        pred=vote.index(max(vote))
    print(vote)
    sign = classes[pred] 
    label.configure(foreground='#011',text=sign)
def classifyr(file_path):
    img = cv2.imread(file_path)
    image = cv2.resize(img, (64, 64))
    image = np.expand_dims(image,axis=0)
    image = np.array(image)
    pred = modelr.predict([image])[0]
    pred = np.argmax(pred)
    print(pred)
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011',text=sign)

def classifya(file_path):
    img = cv2.imread(file_path)
    image = cv2.resize(img, (64, 64))
    image = np.expand_dims(image,axis=0)
    image = np.array(image)
    pred = modela.predict([image])[0]
    pred = np.argmax(pred)
    print(pred)
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011',text=sign)

def classifym(file_path):
    img = cv2.imread(file_path)
    image = cv2.resize(img, (64, 64))
    image = np.expand_dims(image,axis=0)
    image = np.array(image)
    pred = modelm.predict([image])[0]
    pred = np.argmax(pred)
    print(pred)
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011',text=sign)

#initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title("Image Classfication")
top.configure(background= '#CDCDCD')

#set Heading

heading =Label(top, text="Image Classification",pady=20,font=('ariall',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()  


upload = Button(top,text ="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156',foreground="white",font=("arial",10,'bold'))
upload.pack(side = BOTTOM,pady=50)

#uploaded image
sign_image=Label(top)
sign_image.pack(side=BOTTOM,expand=True)

#predicted class
label = Label(top,background="#CDCDCD",font=('arial',15,'bold'))
label.pack(side=BOTTOM,expand=True)

top.mainloop()


