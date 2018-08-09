# -*- coding: utf-8 -*-

'''Toy network for testing this framework.'''

from dango.nodes.pooling import GlobalAveragePool2d, Pool2d
from dango.nodes.container import Branch, Sequential
from dango.nodes.fully_connected import Dense
from dango.networks.basenet import BaseNet
from dango.nodes.convolution import Conv2d
from dango.nodes.upsampler import Deconv2d
from dango.nodes.residual import Residual
from dango.nodes.bn_layer import BNLayer
from dango.nodes.flatten import Flatten
from dango.nodes.match_pool import MatchPool
import tensorflow as tf


class TestNet(BaseNet):
    '''Toy network(must inherit from `BaseNet`).
    
    In such classes, only thing to do is to define `__init__`
    and `forward` functions. Note that the result of `forward`
    should be a dict `logits`. Refer to doc of `BaseNet` for
    more information.
    '''

    def __init__(self, **kwargs):
        super(TestNet, self).__init__(**kwargs)

    def _classification_forward(self):
        with Sequential(name_scope='Net', 
            training=self.training,
            regularizer=self.regularizer) as net:
            net.add(Conv2d(channel=96, filter_strides=2))
            net.add(Pool2d())
            net.add(Conv2d(channel=128))
            net.add(Pool2d())
            net.add(Conv2d(channel=128))
            net.add(Pool2d())
            net.add(Conv2d(channel=128))
            net.add(Pool2d())
            net.add(Flatten())
            net.add(Dense(nodes=128))
            net.add(Dense(nodes=self.num_label['classification'], 
                name_scope='classification-output', 
                with_acti=False))

        logits = {'classification': net(self.input_images)}
        return logits

    def _segmentation_forward(self):
        with Sequential(name_scope='Net0', 
            training=self.training, 
            regularizer=self.regularizer, 
            bn_pre_acti=True) as net:

            net.add(BNLayer(name_scope='seg-bn1'))

            net.add(Conv2d(channel=32, filter_strides=1,
                name_scope='seg-conv1', acti='relu'))
            net.add(MatchPool())
            
            net.add(Conv2d(channel=64, filter_strides=2, 
                name_scope='seg-conv2', acti='relu'))
            net.add(MatchPool())

            net.add(Residual(channel=32, 
                first_block=True, name_scope='seg-res1'))
            net.add(Conv2d(channel=32, acti='relu',
                filter_size=1, name_scope='seg-conv3'))
            net.add(MatchPool())
            
            net.add(Residual(channel=16, 
                with_atrous=True, atrous_rate=2, 
                name_scope='seg-res2'))
            net.add(Conv2d(channel=16, 
                filter_size=1, name_scope='seg-conv4'))
            
            net.add(Residual(with_atrous=True, 
                atrous_rate=4, name_scope='seg-res3'))
            
            net.add(Residual(with_atrous=True, 
                atrous_rate=8, name_scope='seg-res4'))

        with Sequential(name_scope='Net1', 
            training=self.training, 
            regularizer=self.regularizer, 
            bn_pre_acti=True) as net1:

            net1.add(Deconv2d(
                name_scope='seg-deconv',
                output_shape=self.input_shapes['segmentation']+[8])) 
            
            # add one layer, filter_size = 1 ???
            
            net1.add(Conv2d(
                channel=self.num_label['segmentation'],
                filter_size=1, 
                with_acti=False, 
                name_scope='seg-output'))

        logits = {'segmentation': net1(net(self.input_images))}
        return logits

    def _classification_segmentation_forward(self):
        with Branch(name_scope='Net0', 
            training=self.training,
            regularizer=self.regularizer, 
            bn_pre_acti=True) as net:
            net.add(BNLayer(name_scope='seg-bn1'))
            
            # down-pathway
            net.add(Conv2d(channel=32, filter_strides=2, 
                name_scope='seg-conv1'), layer_type='down', decrease_scale=16)
            net.add(Residual(channel=16, first_block=True, name_scope='seg-res1'), 
                layer_type='down', decrease_scale=8)
            net.add(Conv2d(channel=16, filter_size=1, name_scope='seg-conv2'))
            net.add(Residual(channel=8, with_atrous=True, atrous_rate=2, name_scope='seg-res2'), 
                layer_type='down', decrease_scale=4)
            net.add(Conv2d(channel=8, filter_size=1, name_scope='seg-conv3'))
            net.add(Residual(with_atrous=True, atrous_rate=4, name_scope='seg-res3'), 
                layer_type='down', decrease_scale=2)
            net.add(Residual(with_atrous=True, atrous_rate=8, name_scope='seg-res4'), 
                another_task_flag=True)
            
            # up-pathway
            net.add(Conv2d(channel=32))
            net.add(Pool2d(), layer_type='up')
            net.add(Conv2d(channel=64))
            net.add(Pool2d(), layer_type='up')
            net.add(Conv2d(channel=128))
            net.add(Pool2d(), layer_type='up')
            net.add(Conv2d(channel=128))
            net.add(Pool2d(), layer_type='up')
            net.add(Conv2d(channel=128, filter_size=1))

            # classification
            net.add(GlobalAveragePool2d())
            net.add(Flatten())
            net.add(BNLayer())
            net.add(Dense(nodes=64))
            net.add(Dense(nodes=self.num_label['classification'], 
                with_acti=False, name_scope='classification-output'))

        with Sequential(name_scope='Net1', training=self.training, 
            regularizer=self.regularizer, bn_pre_acti=True) as net1:
            net1.add(Deconv2d(name_scope='seg-deconv',
                output_shape=self.input_shapes['segmentation'] + [8]))
            net1.add(Conv2d(channel=self.num_label['segmentation'], 
                filter_size=1, with_acti=False, name_scope='seg-output'))

        logits = dict()
        output_label_image, logits['classification'] = net(self.input_images)
        logits['segmentation'] = net1(output_label_image)

        return logits

    def forward(self):
        _forwards = {
            'classification':
                self._classification_forward,
            'segmentation':
                self._segmentation_forward,
            'classification-segmentation':
                self._classification_segmentation_forward
        }

        return _forwards[self.task]()
