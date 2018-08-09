# -*- coding: utf-8 -*-

'''Flatten.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


class Flatten(BaseLayer):
    '''Flatten conv-results to vectorized form.
    
    attribute: shape:
        shape of flattened data flow.
    '''

    def __init__(self):
        super(Flatten, self).__init__(name_scope='flatten')

    def __call__(self, data_flow):
        '''Flatten.'''
        shape = data_flow.shape.as_list()
        with tf.name_scope(self.name_scope):
            data_flow = tf.reshape(data_flow, [shape[0], -1])
            self.shape = data_flow.shape
        return data_flow