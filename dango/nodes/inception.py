# -*- coding: utf-8 -*-

'''Implementation of inception-block.''' # TODO

from dango.nodes.base_layer import TrainableLayer
from dango.nodes.convolution import Conv2d
from dango.nodes.pooling import Pool2d

import tensorflow as tf

class Inception(TrainableLayer):
    '''Inception-block.'''

    _count = 1

    def __init__(self, channels, acti='relu',
                       regularizer=None,
                       name_scope=None,
                       **kwargs):
        '''
        param: channels: list 
            a list containing channel for each path.
            channels[0] for 1*1 conv-path.
            channels[1:3] for 3*3 conv-path.
            channels[3:5] for 5*5 conv-path.
            channels[5] for pooling-path.
        '''
        if name_scope is None:
            name_scope = 'inception' + str(Inception._count)
            Inception._count += 1
        super(Inception, self).__init__(name_scope=name_scope, **kwargs)

        self.acti = acti
        self.regularizer = regularizer

        self.layers = [
            (self._build_conv_1_path, channels[0], '1'),
            (self._build_conv_3or5_path, channels[1:3], '3'),
            (self._build_conv_3or5_path, channels[3:5], '5'),
            (self._build_pool_path, channels[5], 'pool')
        ]

    def _build_conv_1_path(self, data_flow, channel, name):
        '''Build 1*1 conv-path.'''
        conv = Conv2d(name_scope='conv_1',
                channel=channel, filter_size=1,
                acti=self.acti, trainable=self.trainable,
                regularizer=self.regularizer, info=False)
        conv.variable_scope = self.variable_scope + '-' + conv.name_scope
        data_flow = conv(data_flow)
        
        return data_flow

    def _build_conv_3or5_path(self, data_flow, channel, name):
        '''Build 3*3 conv-path or 5*5 conv-path.'''
        conv1 = Conv2d(name_scope='conv_' + name + '_1',
                channel=channel[0], filter_size=1,
                acti=self.acti, trainable=self.trainable,
                regularizer=self.regularizer, info=False)
        conv1.variable_scope = self.variable_scope + '-' + conv1.name_scope
        conv2 = Conv2d(name_scope='conv_' + name + '_2',
                channel=channel[1], filter_size=int(name),
                acti=self.acti, trainable=self.trainable,
                regularizer=self.regularizer, info=False)
        conv2.variable_scope = self.variable_scope + '-' + conv2.name_scope

        data_flow = conv2(conv1(data_flow))
        return data_flow

    def _build_pool_path(self, data_flow, channel, name):
        '''
        Build pooling-path for this inception-layer.
        '''
        data_flow = Pool2d(filter_strides=1)(data_flow)
        conv =  Conv2d(name_scope='pool-path', 
            channel=channel, filter_size=1, 
            acti=self.acti, trainable=self.trainable,
            regularizer=self.regularizer, info=False)
        conv.variable_scope = self.variable_scope + '-' + conv.name_scope
        data_flow = conv(data_flow)

        return data_flow

    def __call__(self, data_flow):
        '''Feedforward procedure.'''

        data = list()
        with tf.name_scope(self.name_scope):
            for layer in self.layers:
                data.append(layer[0](data_flow, layer[1], layer[2]))
            data_flow = tf.concat(data, axis=3)

        return data_flow
