# -*- coding: utf-8 -*-

'''Optimizer.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


class Optimizer(object):
    '''Base Optimizer.'''

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get_optimization(self, optimizer, loss):
        '''Get optimization operation.
        
        param: loss: tensorflow.Tensor 
            loss to be optimized.
        return: optimization: 
            optimization operation.
        '''
        grads_and_vars = optimizer.compute_gradients(loss)

        for grad, var in grads_and_vars:
            if grad is not None:
                var_name = var.name.rsplit(':', 1)[0]
                BaseLayer.attach_summaries(grad, 
                    '{}-gradients'.format(var_name))
        
        optimization = optimizer.apply_gradients(grads_and_vars, 
            global_step=self.learning_rate.global_step)
        return optimization


class AdamOptimizer(Optimizer):
    '''Adam optimizer.'''

    def __init__(self, learning_rate):
        super(AdamOptimizer, self).__init__(learning_rate)

    def __call__(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate.learning_rate)
            return self.get_optimization(optimizer, loss)


class MomentumOptimizer(Optimizer):
    '''Momentum optimizer.'''

    def __init__(self, learning_rate, momentum=0.9):
        '''
        param: momentum: float 
            momentum.
        '''
        super(MomentumOptimizer, self).__init__(learning_rate)
        self.momentum = momentum

    def __call__(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate.learning_rate,
                momentum=self.momentum)
            return self.get_optimization(optimizer, loss)


class GradientOptimizer(Optimizer):
    '''Gradient optimizer.'''

    def __init__(self, learning_rate):
        super(GradientOptimizer, self).__init__(learning_rate)

    def __call__(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate.learning_rate)
            return self.get_optimization(optimizer, loss)