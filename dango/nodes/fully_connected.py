# -*- coding: utf-8 -*-

'''Fully connected layer, i.e. dense layer.'''

from dango.nodes.bn_layer import BatchNormedActivation
import tensorflow as tf


class Dense(BatchNormedActivation):
    '''Fully connected layer.

    attribute: weight:
        weight of this conv layer.
    attribute: bias:
        bias of this conv layer.
    attribute: shape:
        shape of conved data flow.
    '''

    _fc_counter = 1

    def __init__(self, nodes, 
                       regularizer=None,
                       name_scope=None, 
                       **kwargs):
        '''
        param: nodes: int 
            output nodes.
        param: regularizer: 
            regularizer.
        '''
        if name_scope is None:
            name_scope = 'dense' + str(Dense._fc_counter)
            Dense._fc_counter += 1
        super(Dense, self).__init__(name_scope=name_scope, **kwargs)
        
        self.nodes = nodes
        self.regularizer = regularizer

    def __call__(self, data_flow):
        '''Forward.'''
        input_nodes = data_flow.shape.as_list()[-1]
        with tf.name_scope(self.name_scope):
            self.weight = self.variable_initialiser(name='weight', 
                shape=[input_nodes, self.nodes],
                regularizer=self.regularizer)
            self.bias = self.variable_initialiser(name='bias', 
                shape=[self.nodes], initializer='zeros')

            data_flow = tf.nn.bias_add(
                value=tf.matmul(a=data_flow, b=self.weight),
                bias=self.bias, name='densed')

            data_flow = self.forward(data_flow, self.variable_scope)
            self.shape = data_flow.shape.as_list()

        return data_flow
