#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss
def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob
num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
drop_prob1 = 0.2
drop_prob2 = 0.5
def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3
num_epochs = 5
lr = 0.5
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
