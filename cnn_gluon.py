#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet.gluon import nn
from mxnet import gluon
import sys
sys.path.append('..')
import utils
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation='relu'),
        nn.Dense(10, activation='relu')
    )
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
batch_size = 256
num_epochs = 5
train_data, test_data = utils.load_data_fashion_mnist()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.5})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs)

