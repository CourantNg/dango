# -*- coding: utf-8 -*-

'''Learning rate schedule.'''

import tensorflow as tf


class LearningRate(object):
    '''Naive(non-decayed) learning rate.'''

    def __init__(self, init_rate, name='default'):
        '''
        param: init_rate: float
            initial learning rate.
        param: name: str, optional
            name of this learing rate: 'classification' or
            'segmentation'.
        '''
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = init_rate
        tf.summary.scalar(name + '-lr', self.learning_rate)


class ExponentialDecayedLearningRate(object):
    '''Exponential decayed learning rate.

    decayed_learning_rate = learning_rate *
        decay_rate ^ (global_step / decay_steps)
    '''
    
    def __init__(self, init_rate=1e-4, 
                       decay_steps=2000,
                       decay_rate=0.96, 
                       staircase=True, 
                       name='default'):
        '''
        param: init_rate: float 
            initial learning rate.
        param: decay_steps: int, optional 
            decay steps.
        param: decay_rate: float, optional 
            decay rate.
        param: staircase: bool
            True indicates "global_step / decay_steps" is an 
            integer division.
        param: name: str, optional
            name of this learing rate: 'classification' or
            'segmentation'.
        '''
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=init_rate, global_step=self.global_step,
            decay_steps=decay_steps, decay_rate=decay_rate,
            staircase=staircase)
        tf.summary.scalar(name + '-lr', self.learning_rate)