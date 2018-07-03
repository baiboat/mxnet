#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet.gluon import nn
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Flatten(),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10)

    )
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
net.initialize()
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=64, resize=224)
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})
utils.train(test_iter, test_iter, net, loss, trainer, ctx, num_epochs=1)

