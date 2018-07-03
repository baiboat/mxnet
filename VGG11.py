#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import nn
def vgg_block(num_conv, num_channels):
    blk = nn.Sequential()
    for _ in range(num_conv):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))
def vgg(conv_arch):
    net = nn.Sequential()
    for (num_conv, num_channels) in conv_arch:
        net.add(vgg(num_conv, num_channels))
    net.add(
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10)
    )
    return net
net = vgg(conv_arch)
net.initialize()
net.initialize()
X = nd.random.uniform(shape=(1,1,224,224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
ratio = 4
small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr = 0.05
ctx = gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_data, test_data = gb.load_data_fashion_mnist(batch_size=128, resize=224)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=3)




