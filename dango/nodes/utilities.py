# -*- coding: utf-8 -*-

'''Utilities for Nodes.'''

import tensorflow as tf


class Initializers(object):
    '''Initializers.'''

    @staticmethod
    def Zeros(**kwargs):
        return tf.constant_initializer(0.0)
    
    @staticmethod
    def Ones(**kwargs):
        return tf.constant_initializer(1.0)

    @staticmethod
    def TruncatedNormal(stddev=0.01, **kwargs):
        return tf.truncated_normal_initializer(stddev=stddev)

    @staticmethod
    def XavierNormal(**kwargs):
        return tf.glorot_normal_initializer()

    @staticmethod
    def XavierUniform(**kwargs):
        return tf.glorot_uniform_initializer()

    @staticmethod
    def HeNormal(scale=2.0, mode='fan_in', **kwargs):
        return tf.variance_scaling_initializer(
            scale=scale, mode=mode, distribution='normal')

    @staticmethod
    def HeUniform(scale=2.0, mode='fan_in', **kwargs):
        return tf.variance_scaling_initializer(
            scale=scale, mode=mode, distribution='uniform')


INITIALIZERS = {
    'zeros': Initializers.Zeros,
    'ones': Initializers.Ones,
    'truncated-normal': Initializers.TruncatedNormal,
    'xavier-normal': Initializers.XavierNormal,
    'xavier-uniform': Initializers.XavierUniform,
    'he-normal': Initializers.HeNormal,
    'he-uniform': Initializers.HeUniform
}