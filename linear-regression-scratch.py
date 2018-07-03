#!/usr/bin/env python
#-*- coding:utf-8 -*-
import mxnet.ndarray as nd
import mxnet.autograd as ag
import random
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
batch_size = 10
def data_iter(batch_size, num_examples, features, labels):
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i+batch_size, num_examples)])
        yield features.take(j), labels.take(j)
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w,b]
for param in params:
    param.attach_grad()
def net(X, w, b,):
    return  nd.dot(X, w) + b
def squared_loss(Y_hat, Y):
    return (Y_hat - Y.reshape(Y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
lr = 0.01
epochs = 10
total_loss = 0
for i in range(epochs):
    for X,Y in data_iter(batch_size, num_examples, features, labels):
        with ag.record():
            Y_hat = net(X, w, b)
            loss = squared_loss(Y_hat, Y)
        loss.backward()
        sgd([w,b], lr, batch_size)
    print "epoch %d total_loss is %f"% (i+1,squared_loss(net(features, w, b), labels).mean().asnumpy())
print true_w, w
print true_b, b



