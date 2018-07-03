#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet import gluon
from mxnet import autograd as ag, init
from mxnet import ndarray as nd
import gluonbook as gb
from mxnet.gluon import loss as gloss, nn
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
net = nn.Sequential()
net.add(nn.Flatten())
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})
num_epochs = 5
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)


