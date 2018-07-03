#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
sys.path.append('..')

import gluonbook as gb
from mxnet import ndarray as nd
from mxnet.gluon import loss as gloss
from mxnet import autograd as ag
import utils
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
num_input = 28 * 28
num_output = 10
hidden = 20
scale_param = 0.01
w1 = nd.random.normal(scale=scale_param, shape=(num_input, hidden))
b1 = nd.zeros(shape=(hidden, ))
w2 = nd.random.normal(scale=scale_param, shape=(hidden, num_output))
b2 = nd.zeros(shape=(num_output, ))
params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()
def relu(X):
    return nd.maximum(X, 0)
def net(X):
    X = X.reshape((-1, num_input))
    h1 =  relu(nd.dot(X, w1) + b1)
    output = nd.dot(h1, w2) + b2
    return output
loss = gloss.SoftmaxCrossEntropyLoss()
epochs = 5
learning_rate = 0.01
train_loss = 0.
train_acc = 0.
for i in range(epochs):
    for X,Y in train_iter:
        with ag.record():
            Y_hat = net(X)
            l = loss(Y_hat, Y)
        l.backward()
        utils.SGD(params, learning_rate / batch_size)
    train_loss += nd.mean(l).asscalar()
    train_acc += utils.accuracy(Y_hat, Y)
test_acc = utils.evaluate_accuracy(test_iter, net)
print "Epoch %d train_loss %f train_acc %f test_acc %f" % (i, train_loss/len(train_iter), train_acc/len(train_iter), test_acc)
