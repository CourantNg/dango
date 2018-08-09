# -*- coding: utf-8 -*-

'''MatchPool layer.'''

from dango.nodes.base_layer import BaseLayer
import tensorflow as tf


# import forward `match_pool` and corresponding backward
# procedure, which will be called automatically when
# using `tf.gradients` in `tf.compute_gradients`.
from dango.nodes.new_nodes.match_pool import *


class MatchPool(BaseLayer):
    '''MatchPool layer defined in `new_nodes.match_pool`.'''

    def __init__(self, topk=3,
                       filter_size=3,
                       filter_strides=1,
                       padding='SAME',
                       **kwargs):
        '''
        param: topk: int
            the number of wanted matched features vectors.
        param: filter_size: int
            the size of sliding window.
        param: filter_strides: int  
            the stride of sliding window.
        param: padding: string
            padding type, one of "SAME" and "VALID".
        '''
        super(MatchPool, self).__init__(
            name_scope='match-pooling', **kwargs)
        
        assert isinstance(topk, int), "Need topk to be int."
        assert isinstance(filter_size, int), "Need ksize to be int."
        assert isinstance(filter_strides, int), "Need kstride to be int."
        assert padding in ["SAME", "VALID"], "unrecognized padding type."
        self.topk = topk
        self.ksize = filter_size
        self.kstride = filter_strides
        self.padding = padding

    def __call__(self, data_flow):
        '''Forward.'''
        with tf.name_scope(self.name_scope):
            data_flow, _, _ = match_pool(data_flow, 
                topk=self.topk, ksize=self.ksize, 
                kstride=self.kstride, padding=self.padding)
            tf.summary.histogram('pooled', data_flow)
        return data_flow
