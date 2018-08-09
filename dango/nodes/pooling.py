# -*- coding: utf-8 -*-

'''Pooling.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


class Pool2d(BaseLayer):
    '''2d pooling layer.'''

    def __init__(self, filter_size=2, 
                       filter_strides=2,
                       acti='max', 
                       padding='SAME', 
                       **kwargs):
        '''
        param: filter_size: int or tuple or list 
            filter size.
        param: filter_strides: int, optional 
            filter strides.
        param: padding: str, optional 
            one of 'SAME' or 'VALID'.
        param: acti: str, optional 
            pooling type.
        '''
        if acti not in ['max', 'avg']:
            raise ValueError("unrecognized pooling type.")
        if acti == 'max':
            self.acti = tf.nn.max_pool
            name_scope = 'max-pooling'
        if acti == 'avg':
            self.acti = tf.nn.avg_pool
            name_scope = 'avg-pooling'
        super(Pool2d, self).__init__(name_scope=name_scope, **kwargs)

        if isinstance(filter_size, int):
            self.filter_size = [1, filter_size, filter_size, 1]
        if isinstance(filter_size, (list, tuple)):
            self.filter_size = [1] + list(filter_size) + [1]
        self.strides = [1, filter_strides, filter_strides, 1]
        self.padding = padding

    def __call__(self, data_flow):
        '''Forward.'''
        with tf.name_scope(self.name_scope):
            data_flow = self.acti(value=data_flow,
                ksize=self.filter_size, strides=self.strides,
                padding=self.padding, name='pooled')
            tf.summary.histogram('pooled', data_flow)
        return data_flow


class GlobalAveragePool2d(Pool2d):
    '''2d global average pooling.'''

    def __init__(self):
        super(GlobalAveragePool2d, self).__init__(
            padding='VALID', acti='avg')
        self.name_scope = 'global-average-pooling'

    def __call__(self, data_flow):
        self.filter_size = [1] + data_flow.shape.as_list()[1:-1] + [1] 
        data_flow = super(GlobalAveragePool2d, self).__call__(data_flow)
        return data_flow