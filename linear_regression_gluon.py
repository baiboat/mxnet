#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon import data as gdata
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})
epochs = 5
for epoch in range(epochs):
    for X,y in data_iter:
        with ag.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print("epoch %d, loss: %f"
          % (epoch, loss(net(features), labels).mean().asnumpy()))
dense = net[0]
print true_w, dense.weight.data()
print true_b, dense.bias.data()