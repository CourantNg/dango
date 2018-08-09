# -*- coding: utf-8 -*-

# TODO

from dango.nodes.bn_layer import BatchNormedActivation
from dango.nodes.base_layer import TrainableLayer
from dango.nodes.convolution import Conv2d
from dango.nodes.pooling import Pool2d

import tensorflow as tf

class DenseBlock(TrainableLayer):
    '''Dense-block.'''

    _count = 1

    def __init__(self, layer_num=3,
                       training=None,
                       growth_rate=12,
                       acti='relu',
                       regularizer=None,
                       name_scope=None,
                       **kwargs):
        '''
        param: layer_num: int 
            number of layers in this dense-block.
        param: training: tf.bool(for bn).
        param: growth_rate: int, optional
            growth-rate.
        '''
        if name_scope is None:
            name_scope = 'dense-block' + str(DenseBlock._count)
            DenseBlock._count += 1
        super(DenseBlock, self).__init__(name_scope=name_scope, **kwargs)

        self.layer_num = layer_num
        self.growth_rate = growth_rate

        self.training = training
        self.acti = acti
        self.regularizer = regularizer

    def _build_bn_relu_conv(self, data_flow, name_scope, channel, filter_size):
        '''Build bn-relu-conv block.'''
        data_flow = BatchNormedActivation(acti=self.acti, 
            bn_pre_acti=True, training=self.training,
            trainable=self.trainable)(data_flow)
        conv = Conv2d(name_scope= name_scope + '-conv',
            channel=channel, filter_size=filter_size,
            trainable=self.trainable, with_acti=False,
            regularizer=self.regularizer, info=False)
        conv.variable_scope = self.variable_scope + '-' + conv.name_scope
        data_flow = conv(data_flow)

        return data_flow

    def _build_bottleneck(self, data_flow, name):
        '''Build bottleneck layer.'''
        data_flow = self._build_bn_relu_conv(data_flow, 
            'bottleneck{}-a'.format(name), 4 * self.growth_rate, 1)
        data_flow = self._build_bn_relu_conv(data_flow, 
            'bottleneck{}-b'.format(name), self.growth_rate, 3)
        return data_flow

    def _transition(self, data_flow):
        '''Build transition layer.'''
        data_flow = self._build_bn_relu_conv(data_flow, 'transition',
            self.growth_rate, 1)
        data_flow = Pool2d(acti='avg')(data_flow)
        return data_flow

    def __call__(self, data_flow):
        '''Dense-block procedure.'''
        with tf.name_scope(self.name_scope):
            data_concat = [data_flow]

            data_flow = self._build_bottleneck(data_flow, str(1))
            data_concat.append(data_flow)

            for i in range(1, self.layer_num):
                data_flow = tf.concat(data_concat, axis=3)
                data_flow = self._build_bottleneck(data_flow, str(i + 1))
                data_concat.append(data_flow)

            data_flow = tf.concat(data_concat, axis=3)
            data_flow = self._transition(data_flow)

        return data_flow
