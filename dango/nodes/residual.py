# -*- coding: utf-8 -*-

'''Residual or Mix block.'''

from dango.nodes.bn_layer import BatchNormedActivation
from dango.nodes.base_layer import TrainableLayer
from dango.nodes.convolution import Conv2d
import tensorflow as tf


class Residual(TrainableLayer):
    '''Residual or Mix block.
    
    1) `a`-part of residual block: If it's the first residual block 
    of the network, then this part only consists of conv layer. If 
    size will be decreased, i.e. `size_decrease` is True, then 
    `filter_strides` of this part will be 2. The input argument 
    `filter_size` is also for this part.

    2) `b`-part of residual block: `with_atrous` suggests whether or 
    not to use atrous convolution in this part. Filter size of this 
    part is always 3.

    `channel` is the channel of `residual_data`, and `mix_concat` 
    suggests whether to add input `data_flow` and `residual_data`
    or not.
    '''

    _counter = 1

    def __init__(self, channel=None, 
                       acti='prelu', 
                       filter_size=3, 
                       training=None,
                       first_block=False, 
                       size_decrease=False, 
                       with_atrous=False, 
                       atrous_rate=2,
                       mix_concat=False, 
                       regularizer=None, 
                       name_scope=None, 
                       **kwargs):
        '''
        param: channel: int or None, optional
            channel of middle tensor.
        param: acti: str
            activation function.
        param: filter_size: int
            filter size of first bn-relu-conv.
        param: training: tf.bool
            for batch normalisation.
        param: first_block: bool, optional 
            indicating whether it's the first block.
        param: size_decrease: bool, optional
            indicating whether decreasing the data scale.
        param: with_atrous: bool, optional
            whether using atrous convolution.
        param: atrous_rate: int
            atrous rate.
        param: mix_concat: bool, optional
            indicating whether to concatenate tensor in `mix` way.
        '''
        if name_scope is None:
            name_scope = 'residual' + str(Residual._counter)
            Residual._counter += 1
        super(Residual, self).__init__(name_scope=name_scope, **kwargs)

        self.first_block = first_block
        self.size_decrease = size_decrease

        # if `size_drecrease` is True, then `output_channel` should be
        # as twice as the `input_channel`.
        assert isinstance(channel, int) or channel is None
        if size_decrease: assert channel is None
        self.channel = channel

        self.acti = acti
        self.filter_size = filter_size
        self.training = training
        self.mix_concat = mix_concat
        
        self.with_atrous = with_atrous
        self.atrous_rate = atrous_rate
        self.regularizer = regularizer

    def _build_bn_relu_conv(self, data_flow, 
                                  name_scope, 
                                  filter_strides, 
                                  filter_size, 
                                  with_atrous):
        '''Build bn-relu-conv block.
        
        param: name_scope: str
            name scope of first or second bn-relu-conv, 
            denote by 'a' and 'b' respectively.
        param: filter_strides: int
            stride of conv kernel.
        param: filter_size: int
            filter size of conv kernel.
        param: with_atrous: bool
            whether using atrous convolution(mainly for first block). 
        '''
        variable_scope = '{}-{}-bn-relu'.format(
            self.variable_scope, name_scope)
        bn = BatchNormedActivation(
            acti=self.acti, 
            bn_pre_acti=True, 
            training=self.training, 
            trainable=self.trainable, 
            name_scope='bn-relu')
        data_flow = bn(data_flow, variable_scope)
        
        variable_scope = '{}-{}-conv'.format(
            self.variable_scope, name_scope)
        conv = Conv2d(
            name_scope='conv',
            channel=self.channel, 
            with_acti=False, # no activation
            filter_size=filter_size, 
            filter_strides=filter_strides, 
            with_atrous=with_atrous, 
            atrous_rate=self.atrous_rate,
            regularizer=self.regularizer, 
            trainable=self.trainable,
            variable_scope=variable_scope)
        data_flow = conv(data_flow)

        return data_flow

    def _build_residual_block(self, data_flow, channel):
        '''Build residual block.'''
        strides = 2 if self.size_decrease else 1

        with tf.name_scope('a'):
            if self.first_block:
                variable_scope = '{}-{}-conv'.format(
                    self.variable_scope, 'a')
                conv = Conv2d(
                    name_scope='conv',
                    channel=channel, 
                    with_acti=False, 
                    filter_size=self.filter_size, 
                    filter_strides=strides,
                    with_atrous=False,
                    atrous_rate=self.atrous_rate,
                    regularizer=self.regularizer,
                    trainable=self.trainable,
                    variable_scope=variable_scope)
                data_flow = conv(data_flow)
            else:
                data_flow = self._build_bn_relu_conv(data_flow, 
                    'a', strides, self.filter_size, False)
        
        with tf.name_scope('b'):
            data_flow = self._build_bn_relu_conv(data_flow, 
                'b', 1, 3, with_atrous=self.with_atrous)

        return data_flow

    def __call__(self, data_flow):
        '''Forward.'''
        channel = data_flow.shape.as_list()[-1]
        identity_data = tf.identity(data_flow)

        with tf.name_scope(self.name_scope):
            if self.channel is None: self.channel = channel
            if self.size_decrease: 
                self.channel *= 2
                identity_data = tf.nn.avg_pool(
                    value=identity_data, ksize=[1,2,2,1],
                    strides=[1,2,2,1], padding='SAME')
            self.attach_summaries(identity_data, 'identity-data')

            residual_data = self._build_residual_block(
                data_flow, self.channel)
            self.attach_summaries(residual_data, 'residual-data')

        data_flow = self._add(identity_data, residual_data)
        if self.mix_concat:
            data_flow = tf.concat([data_flow, residual_data], axis=-1)
        self.attach_summaries(data_flow, '{}-data'.format(self.name_scope))

        self.shape = data_flow.shape
        return data_flow