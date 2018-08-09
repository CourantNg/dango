# -*- coding: utf-8 -*-

'''Activation layer.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


class Activate(BaseLayer):
    '''Activation layer.
    
    If this activation layer is under network directly, then
    all operations of this layer should be in this layer's
    name scope.
    '''

    def __init__(self, acti='relu', straight=False):
        '''
        param: acti: str, optional
            activation type.
        param: straight: bool
            indicating whether this layer is under network 
            directly or not.
        '''
        super(Activate, self).__init__(name_scope='activation')
        self.straight = straight

        if acti == 'prelu': self.activate = Activate.prelu
        if acti == 'relu': self.activate = tf.nn.relu
        if acti == 'sigmoid': self.activate = tf.sigmoid
        if acti == 'tanh': self.activate = tf.tanh
        if acti == 'elu': self.activate = tf.nn.elu
    
    @staticmethod
    def prelu(data_flow):
        p = tf.Variable(0.05, name='prelu-parameter')
        tf.summary.scalar('prelu-parameter', p)
        
        positive = tf.nn.relu(data_flow)
        negative = p * (data_flow - tf.abs(data_flow)) * 0.5
        data_flow = positive + negative

        return data_flow

    def __call__(self, data_flow):
        '''Forward.'''
        if self.straight:
            with tf.name_scope(self.name_scope):
                data_flow = self.activate(data_flow)
                tf.summary.histogram('acti', data_flow)
        else:
            data_flow = self.activate(data_flow)
            tf.summary.histogram('acti', data_flow)
        return data_flow