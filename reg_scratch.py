#!/usr/bin/env python
#-*- coding:utf-8 -*-
import  sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
n_train = 20
n_test = 100
num_input = 200
true_w = nd.ones((num_input, 1)) * 0.01
true_b = 0.05
features = nd.random.normal(shape=(n_train + n_test, num_input))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_feature, test_feature = features[:n_train, :], features[n_train:, :]
train_label, test_label = labels[:n_train], labels[n_train:]
def get_params():
    w = nd.random.normal(scale=1, shape=(num_input, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params
def l2_penalty(w):
    return (w**2).sum() / 2
batch_size = 1
num_epochs = 10
lr = 0.003

net = gb.linreg
loss = gb.squared_loss
#InlineBackend.figure_format = 'retina'
gb.plt.rcParams['figure.figsize'] = (3.5, 2.5)
def fit_and_plot(lambd):
    w, b = params = get_params()
    train_ls = []
    test_ls = []
    for _ in range(num_epochs):
        for X, y in gb.data_iter(batch_size, n_train, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            gb.sgd(params, lr, batch_size)
        train_ls.append(loss(net(train_feature, w, b),
                             train_label).mean().asscalar())
        test_ls.append(loss(net(test_feature, w, b),
                            test_label).mean().asscalar())
    gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])
    return 'w[:10]:', w[:10].T, 'b:', b
fit_and_plot(lambd=5)

