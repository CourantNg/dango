# -*- coding: utf-8 -*-

'''Block-Matching layer.'''

from dango.nodes.base_layer import TrainableLayer
from dango.nodes.activation import Activate
from dango.nodes.convolution import Conv2d
import tensorflow as tf


class BMLayer(TrainableLayer):
    '''Block-Matching layer.
    
    If the input of network is BM3D tensor, then this layer should
    be called to deal with data flow.

    attribute: original_channel: int
        channel of input data flow.
    '''
    
    _bm_counter = 1

    def __init__(self, channel,
                       acti='relu',
                       regularizer=None,
                       name_scope=None, 
                       **kwargs):
        '''
        param: channel: int
            data channel of Non-linear transform layer.
        param: acti: str
            activation function. 
        param: regularizer:
            regularizer.
        '''
        if name_scope is None:
            name_scope = 'BM' + str(BMLayer._bm_counter)
            BMLayer._bm_counter += 1
        super(BMLayer, self).__init__(name_scope=name_scope, **kwargs)

        self.channel = channel
        self.acti = acti
        self.regularizer = regularizer

    def _convlayer(self, data_flow, name_scope, channel):
        '''Conv operation in Block Matching layer.

        param: name_scope: str
            name scope of this conv layer.
        param: channel: int
            channel of conved data.
        '''
        variable_scope = self.variable_scope + '-' + name_scope
        conv = Conv2d(
            name_scope=name_scope,
            channel=channel,
            with_acti=False,
            filter_size=1,
            filter_strides=1,
            regularizer=self.regularizer,
            trainable=self.trainable,
            variable_scope=variable_scope)
        data_flow = conv(data_flow)

        return data_flow

    def _nonliner_transform(self, data_flow):
        '''The non-linear transformation in Block-Matching layer.'''
        return Activate(acti=self.acti)(data_flow)

    def __call__(self, data_flow):
        '''Forward of Block-Matching layer.'''
        with tf.name_scope(self.name_scope):
            self.original_channel = data_flow.shape.as_list()[-1]
            data_flow = self._convlayer(data_flow, 'conv1', self.channel)
            data_flow = self._nonliner_transform(data_flow)
            data_flow = self._convlayer(
                data_flow, 'conv2', self.original_channel)
        
        self.shape = data_flow.shape.as_list()
        return data_flow
