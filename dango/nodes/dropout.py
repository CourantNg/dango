# -*- coding: utf-8 -*-

'''Dropout.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


class Dropout(BaseLayer):
    '''Dropout.'''

    def __init__(self, dropout_rate, training):
        '''
        param: dropout_rate: float 
            keep-probability in (0, 1).
        param: training: tf.bool.
            indicating whether or not it's in training phase.
        '''
        super(Dropout, self).__init__(name_scope='dropout')
        self.dropout_rate = dropout_rate
        self.training = training

    def __call__(self, data_flow):
        '''Forward.'''
        with tf.name_scope(self.name_scope):
            def true_fun(): 
                return tf.nn.dropout(data_flow, self.dropout_rate)
            def false_fun(): 
                return data_flow
            data_flow = tf.cond(self.training, true_fun, false_fun)
        return data_flow