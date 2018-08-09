# -*- coding: utf-8 -*-

'''Upsampler: Deconv2d.'''

from dango.nodes.bn_layer import BatchNormedActivation
from dango.nodes.convolution import Conv2d
import tensorflow as tf


class Deconv2d(BatchNormedActivation):
    '''Deconvolution upsampler.
    
    If both input arguments `output_shape` and `variable_scope` are 
    Conv2d instances, then these two instances are successive Conv2d 
    instances `conv1` and `conv2`. Attributes `output_shape` and 
    `variable_scope` can be obtained by `conv1` and `conv2` respectively.

    attribute: output_shape:
        output_shape of this deconv layer.  
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

    _deconv_counter = 1

    def __init__(self, output_shape=None,
                       filter_size=3, 
                       filter_strides=2, 
                       padding='SAME',
                       with_atrous=False, 
                       atrous_rate=2,
                       regularizer=None, 
                       variable_scope=None,
                       name_scope=None, 
                       **kwargs):
        '''
        param: output_shape: list, tuple or Conv2d
            output shape of this deconv layer.
        param: filter_strides: int, optional
            filter strides, the same as corresponding conv operation,
            can be obtained by Conv2d `_variable_scope`.
        param: filter_size: int or list or tuple
            filter size, the same as corresponding conv operation,
            can be obtained by Conv2d `_variable_scope`.
        param: padding: str, optional
            'SAME' or 'VALID'.
        param: with_atrous: bool, optional
            whether using atrous convolution.
        param: atrous_rate: int, optional
            atrous rate, the same as `filter_strides` generally.
        param: variable_scope: str or Conv2d
            variable scope.
        param: regularizer:
            regularizer.
        '''
        if name_scope is None:
            name_scope = 'deconv' + str(Deconv2d._deconv_counter)
            Deconv2d._deconv_counter += 1
        super(Deconv2d, self).__init__(name_scope=name_scope, **kwargs)

        self.strides = [1, filter_strides, filter_strides, 1]
        assert padding in ['SAME', 'VALID']
        self.padding = padding 

        if isinstance(filter_size, int): 
            self.filter_size = [filter_size, filter_size]
        elif isinstance(filter_size, (list, tuple)):
            self.filter_size = list(filter_size)
        else: raise TypeError("'filter_size' should be int, list or tuple.")

        self.with_atrous = with_atrous
        self.atrous_rate = atrous_rate
        self.regularizer = regularizer

        assert isinstance(output_shape, (Conv2d, list, tuple))
        self.output_shape = output_shape

        assert (isinstance(variable_scope, (Conv2d, str))
            or variable_scope is None)
        self._variable_scope = variable_scope

    def __call__(self, data_flow):
        '''Forward.'''
        # output shape
        if isinstance(self.output_shape, Conv2d):
            self.output_shape = self.output_shape.shape
        if isinstance(self.output_shape, tuple):
            self.output_shape = list(self.output_shape)
        output_channel = self.output_shape[-1]
        self.filter_size += [output_channel, data_flow.shape[-1]]

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
            self.strides = self._variable_scope.strides
            self.filter_size = self._variable_scope.filter_size

        with tf.name_scope(self.name_scope):
            self.weight = self.variable_initialiser(name='weight', 
                shape=self.filter_size, regularizer=self.regularizer)
            self.bias = self.variable_initialiser(name='bias', 
                shape=[output_channel], initializer='zeros')
            
            if self.with_atrous:
                data_flow = tf.nn.atrous_conv2d_transpose(
                    value=data_flow, filters=self.weight,
                    output_shape=self.output_shape,
                    rate=self.atrous_rate, padding=self.padding)
            else:
                data_flow = tf.nn.conv2d_transpose(
                    value=data_flow, filter=self.weight,
                    output_shape=self.output_shape,
                    strides=self.strides, padding=self.padding)
            
            data_flow = tf.nn.bias_add(value=data_flow, 
                bias=self.bias, name='deconved')
                
            data_flow = self.forward(data_flow, variable_scope) 
            self.shape = data_flow.shape.as_list()

        return data_flow
