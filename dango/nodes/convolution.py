# -*- coding: utf-8 -*-

'''Convolution layers.'''

from dango.nodes.bn_layer import BatchNormedActivation
import tensorflow as tf


class Conv2d(BatchNormedActivation):
    '''2d convolutional layer.
    
    attribute: variable_scope:
        this desired attribute is assigned as `container-layer`
        when calling this instance to deal with data flow.
    attribute: _variable_scope:
        receive value of input argument `variable_scope` as 
        a condidate value of the attribute `variable_scope`
        defined in `TrainableLayer.__init__`. Refer to doc of
        `dango.nodes.base_layer.BaseLayer` for more information.
        If `_variable_scope` is a Conv2d instance, which must 
        must have dealt with data flow already, then `variable_scope`
        can be assigned as this instance's `variable_scope`. 
    attribute: weight:
        weight of this conv layer.
    attribute: bias:
        bias of this conv layer.
    attribute: shape:
        shape of conved data flow.
    '''

    _conv2d_counter = 1

    def __init__(self, channel, 
                       filter_size=3, 
                       filter_strides=1,
                       padding='SAME',
                       with_atrous=False, 
                       atrous_rate=2, 
                       regularizer=None, 
                       variable_scope=None,
                       name_scope=None, 
                       **kwargs):
        '''
        param: channel: int 
            channel of output tensor.
        param: filter_size: int or list or tuple, optional 
            filter size.
        param: filter_strides: int 
            filter strides.
        param: padding: str 
            padding type, one of 'SAME' and 'VALID'. 
        param: with_atrous: bool, optional
            whether using atrous convolution.
        param: atrous_rate: int
            atrous rate. 
        param: regularizer:
            regularizer.
        '''
        if name_scope is None:
            name_scope = 'conv' + str(Conv2d._conv2d_counter)
            Conv2d._conv2d_counter += 1
        super(Conv2d, self).__init__(name_scope=name_scope, **kwargs)

        self.channel = channel
        assert padding in ['SAME', 'VALID']
        self.padding = padding
        self.strides = [1, filter_strides, filter_strides, 1]

        if isinstance(filter_size, int):
            self.filter_size = [filter_size] * 2
        elif isinstance(filter_size, (list, tuple)):
            self.filter_size = list(filter_size)
        else: raise TypeError("'filter_size' should be int, list or tuple.")

        self.with_atrous = with_atrous
        self.atrous_rate = atrous_rate
        self.regularizer = regularizer

        assert (isinstance(variable_scope, (Conv2d, str))
            or variable_scope is None)        
        self._variable_scope = variable_scope
    
    def __call__(self, data_flow):
        '''Forward.'''
        input_channel = data_flow.shape.as_list()[-1]

        # variable scope
        # 
        # `BatchNormedActivation.forward` will be called below, and
        # one argument is `variable_scope` which should differ from 
        # the variable scope of Conv2d `_variable_scope`(so that
        # parameters of batch normlisation will be different).
        variable_scope = self.variable_scope
        if isinstance(self._variable_scope, str):
            self.variable_scope = self._variable_scope
        if isinstance(self._variable_scope, Conv2d):
            self.variable_scope = self._variable_scope.variable_scope

        with tf.name_scope(self.name_scope):
            self.weight = self.variable_initialiser(name='weight',
                shape=self.filter_size + [input_channel, self.channel],
                regularizer=self.regularizer)
            self.bias = self.variable_initialiser(name='bias', 
                shape=[self.channel], initializer='zeros')

            if self.with_atrous:
                data_flow = tf.nn.atrous_conv2d(value=data_flow, 
                    filters=self.weight, rate=self.atrous_rate, 
                    padding=self.padding)
            else:
                data_flow = tf.nn.conv2d(input=data_flow, 
                    filter=self.weight, strides=self.strides, 
                    padding=self.padding)

            data_flow = tf.nn.bias_add(value=data_flow, 
                bias=self.bias, name='conved')
            
            data_flow = self.forward(data_flow, variable_scope)
            self.shape = data_flow.shape.as_list()
        
        return data_flow