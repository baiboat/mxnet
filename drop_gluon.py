#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
sys.path.append("..")
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
drop_prob1 = 0.2
drop_prob2 = 0.5
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Dense(256, activation='relu'),
        nn.Dropout((drop_prob1)),
        nn.Dense(256, activation='relu'),
        nn.Dropout((drop_prob2)),
        nn.Dense(10)
    )
net.initialize(init.Normal(sigma=0.01))
num_epochs = 5
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist()
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
