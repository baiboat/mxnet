#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet.gluon import nn
def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1, strides=strides, padding=0, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1, strides=strides, padding=0, activation='relu'),
    )
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out
net = nn.Sequential()
with net.name_scope():
    net.add(
        mlpconv(channels=96, kernel_size=11, padding=0, strides=4),
        mlpconv(channels=256, kernel_size=5, padding=2),
        mlpconv(channels=384, kernel_size=3, padding=1),
        nn.Dropout(.5),
        mlpconv(10, 3, 1, max_pooling=False),
        nn.GlobalAvgPool2D(),
        nn.Flatten()
    )
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon
from mxnet import init
train_data, test_data = gb.load_data_fashion_mnist(batch_size=64, resize=224)
ctx = gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})
gb.train(train_data, test_data, net, loss, trainer, ctx, 5)