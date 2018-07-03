#!/usr/bin/env python
#-*- coding:utf-8 -*-
#%matplotlib inline
import sys
sys.path.insert(0, '..')
import gluonbook as gb
from mxnet import nd, image, gluon, init
from mxnet.gluon.data.vision import transforms
img = image.imread('G:/gluon/mxnet/img/cat1.jpg')
gb.plt.imshow(img.asnumpy())
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_cols*num_rows)]
    gb.show_images(Y, num_rows, num_cols,scale)
apply(img, transforms.RandomFlipLeftRight())
shape_aug = transforms.RandomResizedCrop((200, 200), scale=(.1, 1), ratio=(.5, 2))
apply(img, shape_aug)
apply(img, transforms.RandomLighting(.5))
apply(img, transforms.RandomHue(.5))
train_augs = transforms.Compose([
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
])

test_augs = transforms.Compose([
    transforms.ToTensor(),
])
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(gluon.data.vision.CIFAR10(
        train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=2)
def train(train_augs, test_augs, lr=.1):
    batch_size = 256
    ctx = gb.try_all_gpus()
    net = gb.resnet18(10)
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train_data = load_cifar10(True, train_augs, batch_size)
    test_data = load_cifar10(False, test_augs, batch_size)
    gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=8)
train(train_augs, test_augs)