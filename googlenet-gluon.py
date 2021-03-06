#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet.gluon import nn
from mxnet import nd
class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_conv_1 = nn.Conv2D(n1_1, kernerl_size=1, activation='relu')
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')
    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(p1))
        p3 = self.p3_conv_5(self.p3_conv_1(p2))
        p4 = self.p4_conv_1(self.p4_pool_3(p3))
        return nd.concat(p1, p2, p3, p4, dim=1)
class GoogLeNet(nn.Block):
    def __init__(self, num_class, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu')
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(128, 128, 191, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )
            b6 = nn.Sequential()
            b6.add(
                nn.Flatten(),
                nn.Dense(num_classes)
            )
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)
    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon
from mxnet import init
train_data, test_data = gb.load_data_fashion_mnist(batch_size=64, resize=960)
ctx = gb.try_gpu()
net = GoogLeNet(10)
net.initialize(ctx=ctx, init=init.Xavier())
loss = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)