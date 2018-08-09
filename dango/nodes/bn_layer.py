# -*- coding: utf-8 -*-

'''Batch normalisation.'''

from dango.nodes.base_layer import TrainableLayer
from dango.nodes.activation import Activate
import tensorflow as tf


class BNLayer(TrainableLayer):
    '''Batch normalisation layer.

    attribute: variable_scope:
        this desired attribute is assigned as `container-layer`
        when calling this instance to deal with data flow.
    attribute: _variable_scope:
        receive value of input argument `variable_scope` as 
        a condidate value of the attribute `variable_scope`
        defined in `TrainableLayer.__init__`. Refer to doc of
        `dango.nodes.base_layer.BaseLayer` for more information.
    '''

    _bn_counter = 1

    def __init__(self, training=None, 
                       moving_decay=0.99, 
                       name_scope=None, 
                       variable_scope=None, 
                       straight=True, 
                       **kwargs):
        '''
        param: training: tf.bool
            indicating whether or not bn layer is in training phase.
        param: moving_decay: float
            moving average decay rate.
        param: straight: bool
            indicating whether this layer is under network directly
            or not.
        '''
        if name_scope is None:
            name_scope = 'bn{}'.format(BNLayer._bn_counter)
            BNLayer._bn_counter += 1
        super(BNLayer, self).__init__(name_scope=name_scope, **kwargs)

        self.training = training
        self.moving_decay = moving_decay
        self.straight = straight

        # variable scope
        assert isinstance(variable_scope, str) or variable_scope is None
        self._variable_scope = variable_scope

    def _batch_norm(self, data_flow, axes, channels):
        '''The main procedure of batch normalisation layer.'''
        beta = self.variable_initialiser(name='beta', 
            shape=[channels], initializer='zeros')
        gamma = self.variable_initialiser(name='gamma', 
            shape=[channels], initializer='ones')

        batch_mean, batch_var = tf.nn.moments(data_flow, axes)
        ema = tf.train.ExponentialMovingAverage(decay=self.moving_decay)
        def param_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        def param_apply():
            return ema.average(batch_mean), ema.average(batch_var)
        mean, var = tf.cond(self.training, param_update, param_apply)

        data_flow = tf.nn.batch_normalization(
            x=data_flow, mean=mean, variance=var,
            offset=beta, scale=gamma, variance_epsilon=1e-3)
        return data_flow

    def __call__(self, data_flow):
        '''Forward procedure of bn layer.
        
        If this bn layer is a layer of network directly, not a sub-layer
        of a layer of network(e.g. conv layer), then these operations in 
        this layer should be aggregated in this layer's name scope. Also, 
        the histogram of data flow after batch normalisation, in this case,
        should be put in tensorboard for visualization.  
        '''
        if isinstance(self._variable_scope, str):
            self.variable_scope = self._variable_scope
            
        assert self.training is not None
        shape = data_flow.shape.as_list()
        channels = shape[-1]
        axes = list(range(len(shape)))[:-1]

        if self.straight:
            with tf.name_scope(self.name_scope):
                data_flow = self._batch_norm(data_flow, axes, channels)
                tf.summary.histogram('bn', data_flow)
        else:
            data_flow = self._batch_norm(data_flow, axes, channels)

        return data_flow


class BatchNormedActivation(TrainableLayer):
    '''An auxiliary sub-layer: activation with batch normalisation.'''

    def __init__(self, with_acti=True, 
                       acti='prelu',
                       bn_pre_acti=False, 
                       bn_after_acti=False, 
                       training=None, 
                       moving_decay=0.9, 
                       name_scope=None, 
                       **kwargs):
        '''
        param: with_acti: bool, optional
            indicating whether data flow will be activated.
        param: acti: str, optional
            activation function.
        param: bn_pre_acti: bool, optional
            whether using batch normalisation before activation.
        param: bn_after_acti: bool, optional
            whether using batch normalisation after activation.
        '''
        assert name_scope is not None
        super(BatchNormedActivation, self).__init__(
            name_scope=name_scope, **kwargs)
        self.kwargs = kwargs # 'trainable' may be used in 'forward'

        # activation
        self.with_acti = with_acti
        self.acti = acti

        # batch-normalisation
        self.bn_pre_acti = bn_pre_acti
        self.bn_after_acti = bn_after_acti
        self.training = training
        self.moving_decay = moving_decay
        if all([self.bn_pre_acti, self.bn_after_acti]):
            raise ValueError("batch normalisation should not be used "
                "pre and after activation in one layer at the same time.")

    def forward(self, data_flow, variable_scope):
        '''Forward.'''
        tf.summary.histogram('pre-acti', data_flow)
        name_scope = self.name_scope + '-bn'

        if self.bn_pre_acti:
            data_flow = BNLayer(self.training, self.moving_decay,
                name_scope, variable_scope, False, **self.kwargs)(data_flow)
            tf.summary.histogram('bn-pre-acti', data_flow)

        if self.with_acti: data_flow = Activate(self.acti)(data_flow)

        if self.bn_after_acti:
            data_flow = BNLayer(self.training, self.moving_decay, 
                name_scope, variable_scope, False, **self.kwargs)(data_flow)
            tf.summary.histogram('bn-after-acti', data_flow)

        return data_flow

    def __call__(self, data_flow, variable_scope):
        return self.forward(data_flow, variable_scope)