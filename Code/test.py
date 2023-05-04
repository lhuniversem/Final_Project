import tensorflow as tf
from tensorflow import keras
# 定义一个basic block
class BasicBlock(keras.layers.Layer):
    def __init__(self,input_filters,filters,stride = 1):
        super(BasicBlock,self).__init__()
        self.conv1 = keras.layers.Conv2D(filters,[3,3],strides=stride,padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation("relu")
        self.conv2 = keras.layers.Conv2D(filters,[3,3],strides = 1,padding = "same")
        self.bn2 = keras.layers.BatchNormalization()
        if stride != 1 or input_filters != filters:
            self.subsampling = keras.Sequential([
                                                keras.layers.Conv2D(filters,[1,1],strides = stride,padding = "same"),\
                                                keras.layers.BatchNormalization()
            ])
        else:
            self.subsampling = lambda x:x
    
    def call(self,inputs,training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.subsampling(inputs)
        out = keras.layers.add([out,residual])
        outputs = self.relu(out)
        return outputs
# 定义一个ResNet Block
def ResNetBlock(num_block,input_filters,filters,stride = 1):
    resnet = keras.Sequential()
    resnet.add(BasicBlock(input_filters,filters,stride = stride))
    for i in range(1,num_block):
        resnet.add(BasicBlock(filters,filters,stride = 1))
    return resnet
# 定义一个 ResNet （模型网络）
class ResNet(keras.Model):
    def __init__(self,n_classes = 100,block_dim = [2,2,2,2]):
        super(ResNet,self).__init__()
        self.root = keras.Sequential([keras.layers.Conv2D(64,[3,3],strides=[1,1],padding = "same"),\
                                      keras.layers.BatchNormalization(),
                                      keras.layers.MaxPool2D(pool_size=(2,2),strides = 1)
                                     ])                       
        self.layer1 = ResNetBlock(block_dim[0],64,64,stride=2)
        self.layer2 = ResNetBlock(block_dim[1],64,128,stride = 2)
        self.layer3 = ResNetBlock(block_dim[2],128,256,stride = 2)
        self.layer4 = ResNetBlock(block_dim[3],256,512,stride = 2)
        # output = [b,h,w,512] => [b,c]
        self.avgpool = keras.layers.GlobalAvgPool2D()
        self.fc = keras.layers.Dense(n_classes)
        
    def call(self,inputs,training = None):
        out = self.root(inputs)
        out = self.layer1(out)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        outputs = self.fc(out)
        
        return outputs
# 搭建18层的resnet网络
def resnet():
    return ResNet(100,[2,2,2,2])
# 预处理函数
def preprocess(x,y):
    x = 2*(tf.cast(x,dtype=tf.float32)/255.-0.5)
    y = tf.cast(y,dtype=tf.int32)
    y_onehot = tf.one_hot(y,depth=100)
    return x,y_onehot
# 加载数据集
(x_train,y_train),(x_test,y_test) = keras.datasets.mri.load_data()#keras.datasets.cifar100.load_data()
print(type(x_train),type(y_train))
y_train,y_test = tf.squeeze(y_train,axis=1),tf.squeeze(y_test,axis=1)
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_train = db_train.map(preprocess).shuffle(100000).batch(32)
db_test = db_test.map(preprocess).shuffle(100000).batch(32)
sample = next(iter(db_train))
sample[1].shape
# 创建模型
res_net18 = resnet() 
res_net18.compile(loss = tf.losses.CategoricalCrossentropy(),\
                 optimizer = keras.optimizers.Adam(learning_rate = 1e-4),\
                  metrics = ["accuracy"]
                 )
res_net18.fit(db_train,epochs = 10,validation_data = db_test,validation_freq = 1)