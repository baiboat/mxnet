#!/usr/bin/env python
#-*- coding:utf-8 -*-
import mxnet.ndarray as nd
import mxnet.autograd as ag
import random
num_examples = 1000
num_input = 2
ture_w = [2, -3.4]
ture_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_input))
labels = ture_w[0] * features[:, 0] + ture_w[1] * features[:, 1] + ture_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
batch_size = 10
def data_iter(num_examples, batch_size, features, labels):
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index(i:i+batch_size)
        yield features.take(j), labels.take(j)
w = nd.random.normal(scale=1, shape=(1,num_input))
b = nd.zeros(shape=(1, ))
param = [w,b]

def net(X, w, b):
    return nd.dot(X, w) + b
def squred_loss(Y_hat, Y):
    return (Y_hat - Y) ** 2 / 2
def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
lr = 0.01
epochs = 5
total_loss = 0
for i in range(epochs):
    for X,Y in data_iter(num_examples, batch_size, features, labels)
        with ag.record():
            Y_hat = net(X, w, b)
            loss = squred_loss(Y_hat, Y)
        loss.backward()
        sgd([w,b], lr)
    print "epoch %d loss is %f"%(i,loss(net(features, w, b)).means().asnumpy())


