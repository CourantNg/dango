# -*- coding: utf-8 -*-

'''Regularizer.'''

import tensorflow as tf


def l1_regularizer(_lambda=1e-5):
    return tf.contrib.layers.l1_regularizer(_lambda)


def l2_regularizer(_lambda=1e-5):
    return tf.contrib.layers.l2_regularizer(_lambda)