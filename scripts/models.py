import numpy as np
import os
import cv2

import chainer
import chainercv
from chainer import functions as F
from chainer import links as L

from functools import partial


def backborn_provider(backborn='resnet50', pretrained_model='imagenet'):
    if backborn == 'resnet50':
        return L.ResNet50Layers(pretrained_model=pretrained_model)
    elif backborn == 'resnet101':
        return L.ResNet101Layers(pretrained_model=pretrained_model)
    elif backborn == 'resnet152':
        return L.ResNet152Layers(pretrained_model=pretrained_model)
    elif backborn == 'seresnext50':
        return chainercv.links.model.senet.SEResNeXt50(pretrained_model=pretrained_model)
    elif backborn == 'seresnext101':
        return chainercv.links.model.senet.SEResNeXt101(pretrained_model=pretrained_model)
    else:
        raise Exception(
            'NotImplementedError: This is an unimplemented model name.')


class ConvBNR(chainer.Chain):
    def __init__(self, ch0, ch1, use_bn=True,
                 sample='down', activation=F.relu, dropout=False):
        self.use_bn = use_bn
        self.activation = activation
        self.dropout = dropout
        w = chainer.initializers.Normal(0.02)
        self.sample = sample
        super(ConvBNR, self).__init__()
        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
#                 self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            if use_bn:
                self.bn = L.BatchNormalization(ch1)

    def forward(self, x):
        h = self.c(x)
        if self.sample == 'up':
            h = F.unpooling_2d(h, 4, 2, 1, outsize=(
                h.shape[2]*2, h.shape[3]*2))
        if self.use_bn:
            h = self.bn(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class ResUNet(chainer.Chain):
    '''
    resnet layers output
    conv1 -> (1, 64, H/2, W/2)
    pool1 -> (1, 64, H/4, W/4)
    res2 -> (1, 256, H/4, W/4)
    res3 -> (1, 512, H/8, W/8)
    res4 -> (1, 1024, H/16, W/16)
    res5 -> (1, 2048, H/32, W/32)
    '''

    # 'reduce_param' divides the number of output layers during convolution.
    def __init__(self, n_class, return_hidden=False, backborn='resnet50', pretrained_model='auto', reduce_param=1, color=True):
        w = chainer.initializers.HeNormal()
        super(ResUNet, self).__init__()
        self.layers = ['conv1', 'pool1', 'res2', 'res3', 'res4', 'res5']
        self.color = color
        self.n_class = n_class
        out_arr = [64, 64, 256, 512, 1024, 2048]
        in_arr = [32, 64, 64, 256, 512, 1024]
        self.return_hidden = return_hidden
        with self.init_scope():
            if self.color == False:
                self.first_layer = L.Convolution2D(
                    None, 3, initialW=w, ksize=1, stride=1, pad=0)  # gray
            self.base_model = backborn_provider(
                backborn=backborn, pretrained_model=pretrained_model)

            for i, layer in enumerate(self.layers):
                if layer[:-1] == 'pool':
                    setattr(self, layer, partial(
                        F.average_pooling_2d, ksize=4, stride=2, pad=1))
                else:
                    setattr(self, layer, self.base_model[layer])
                setattr(self, layer+'_1x1', L.Convolution2D(None,
                                                            int(out_arr[i] / reduce_param), initialW=w, ksize=1, stride=1, pad=0))
                if self.layers[i-1][:-1] == 'pool':
                    setattr(self, layer+'_up', ConvBNR(None,
                                                       int(in_arr[i] / reduce_param), use_bn=True, sample='mid', dropout=True))
                else:
                    setattr(self, layer+'_up', ConvBNR(None,
                                                       int(in_arr[i] / reduce_param), use_bn=True, sample='up', dropout=True))
            self.conv_last = L.Convolution2D(
                None, self.n_class, initialW=w, ksize=1, stride=1, pad=0)

    def forward(self, x):
        if self.color == False:
            x = self.first_layer(x)
        hs = [x]

        for i, layer in enumerate(self.layers):
            hs.append(self[layer](hs[i]))
        h = None
        for i, layer in enumerate(self.layers[::-1]):
            skip = F.relu(self[layer+'_1x1'](hs[-i-1]))
            if i == 0:
                h = self[layer+'_up'](skip)
            else:
                h = F.concat([h, skip])
                h = self[layer+'_up'](h)

        result = F.sigmoid(self.conv_last(h))
        if self.return_hidden:
            return result, hs[-1]
        else:
            return result

    def predict(self, x):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            result, _ = self.forward(x)


if __name__ == '__main__':
    # simple test
    net = ResUNet(return_hidden=True, n_class=4, backborn='resnet50',
                  pretrained_model='auto', reduce_param=1)
#     reg_net = RegMLP()
    x = np.zeros((1, 3, 256, 256)).astype('f')
    condition = np.zeros((1, 2, 8, 8)).astype('f')
    result = net(x, condition)
#     result_reg = reg_net(h, condition)
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(net)
    for func_name in net.base_model._children:
        for param in net.base_model[func_name].params():
            param.update_rule.hyperparam.alpha *= 0.1
    t = np.zeros((1, 2, 256, 256)).astype('i')
    loss = F.sigmoid_cross_entropy(result, t)
    print(loss)
    assert (1, 4, 256, 256) == result.shape
    print(result.shape, result_reg.shape)
