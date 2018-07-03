#!/usr/bin/env python
#-*- coding:utf-8 -*-
from mxnet import nd
from mxnet.gluon import nn
def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out
class DenseBlock(nn.Block):
    def __init__(self, layer, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layer):
            self.net.add(conv_block(growth_rate))
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out
init_channels = 64
growth_rate = 32
blaock_layers = [6, 12, 24, 16]
num_class = 10
def dense_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(in_channels, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                net.add(transition_block(channels//2))

        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_class)
        )
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon
from mxnet import init

train_data, test_data = gb.load_data_fashion_mnist(
    batch_size=64, resize=32)

ctx = gb.try_gpu()
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)