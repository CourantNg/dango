# -*- coding: utf-8 -*-

'''Containers including Sequential and Branch.'''

from dango.nodes.base_layer import BaseLayer
from dango.nodes.pooling import Pool2d
import tensorflow as tf


class Sequential(object):
    '''Sequential containers.
    
    attribute: layers: list
        containing layer instances.
    attribute: collections: list
        collected info of this sequential.
    '''

    _sequential_counter = 1

    def __init__(self, name_scope=None, **kwargs):
        '''
        param: name_scope: str or None, optional
            name scope of this sequential.
        '''
        if name_scope is None:
            name_scope = 'sequential' + str(Sequential._sequential_counter)
            Sequential._sequential_counter += 1
        self.name_scope = name_scope

        self.layers, self.collections = list(), list()
        self.kwargs = kwargs

    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        for layer in self.layers:
            layer.variable_scope = self.name_scope + '-' + layer.name_scope
            for key, value in self.kwargs.items():
                if hasattr(layer, key): setattr(layer, key, value)

    def add(self, layer):
        '''Add layer to this container.
        
        param: layer: 
            layer instance.
        return: layer
            same layer instance.
        '''
        self.layers.append(layer)
        return layer

    def _logger(self):
        '''Logging of this sequential.'''
        messages = ['[{}]'.format(self.name_scope)]
        for layer_info in self.collections:
            messages.append('--[name_scope]: {}'.format(
                layer_info['name_scope']))
            for key, value in layer_info.items():
                if key != 'name_scope':
                    messages.append('----{}: {}'.format(
                        key, value))
        tf.logging.info('network(s) info'.center(100, '*'))
        [tf.logging.info(msg) for msg in messages]

    def __call__(self, data_flow):
        '''Forward.'''
        with tf.name_scope(self.name_scope):
            for layer in self.layers:
                data_flow = layer(data_flow)
                
                fail_msg = "output of {}-{} is not finite".format(
                    self.name_scope, layer.name_scope)
                data_flow = tf.verify_tensor_all_finite(
                    data_flow, fail_msg)
                
                self.collections.append(layer.__dict__)

        self._logger()
        return data_flow


class Branch(Sequential):
    '''Branch containers.

    attribute: pairs: list
        an intermediate list to generate `down_path` and `up_path`.
    attribute: down_path: dict
        key   <--> index of node in down path.
        value <--> some properties will be used in connection.
    attribute: up_path: dict
        key   <--> index of node in up path.
        value <--> index of connection node in down path.
    '''
    
    _branch_counter = 1 
    _layer_types = ['down', 'up']
    _connection = ['add', 'concat']

    def __init__(self, name_scope=None, **kwargs):
        if name_scope is None:
            name_scope = 'branch' + str(Branch._branch_counter)
            Branch._branch_counter += 1
        super(Branch, self).__init__(name_scope=name_scope, **kwargs)
        
        self.pairs = list()
        self.down_path, self.up_path = dict(), dict()

    def add(self, layer, 
                  layer_type=None, 
                  decrease_scale=1, 
                  connection='concat', 
                  another_task_flag=False):
        '''Add layer to this container.

        NOTE: how to use `down` and `up` as `layer_type`? 
        1) the 'layer' which will generate data that have two branches should 
        be equipped with 'down' layer-type;
        2) the 'layer' which will generate data that will be concated by other
        data should be equipped with 'up' layer-type.

        param: layer:
            layer instance.
        param: layer_type: str or None, optional
            should be one of Branch._layer_types or None.
        param: decrease_scale: int, optional
            scale for decreasing size, default is 1, i.e. no decreasing.
        param: connection: str, optional
            the way to combine two tensors, one of Branch_connection.
        param: another_task_flag: bool, optional
            indicating the index of layer which will generate data as input 
            of another task.
        return: layer: 
            layer instance.
        '''
        self.layers.append(layer)
        num = len(self.layers) - 1

        if layer_type:
            assert layer_type in Branch._layer_types
            
            if layer_type == 'down':
                assert connection in Branch._connection
                self.pairs.append((num, decrease_scale, connection))
            if layer_type == 'up':
                (_num, _decrease_scale, _connection) = self.pairs.pop()
                self.down_path[_num] = (_decrease_scale, _connection)
                self.up_path[num] = _num

        if another_task_flag:
            self.another_task_index = num

        return layer

    def _connect_branches(self, data1, data2, decrease_scale, connection):
        '''Branch connection.'''
        if decrease_scale != 1:
            data21 = Pool2d(acti='avg', filter_size=decrease_scale, 
                filter_strides=decrease_scale)(data2)
            data22 = Pool2d(acti='max', filter_size=decrease_scale, 
                filter_strides=decrease_scale)(data2)
            data2 = tf.concat([data21, data22], axis=-1)

        if connection == 'add':
            return BaseLayer._add(data1, data2)
        if connection == 'concat':
            return tf.concat([data1, data2], axis=-1)

    def __call__(self, data_flow):
        '''Forward.'''
        assert len(self.pairs) == 0, "down- and up-path are not sysmetric."
        branch_data = dict() # for storing data to be connected
        
        with tf.name_scope(self.name_scope):
            for index, layer in enumerate(self.layers):
                data_flow = layer(data_flow)
                self.collections.append(layer.__dict__)
                
                if index in self.down_path.keys():
                    branch_data[index] = {
                        'data_flow': data_flow, 
                        'decrease_scale': self.down_path[index][0],
                        'connection': self.down_path[index][1]
                    }
                if index in self.up_path.keys():
                    _key = self.up_path[index]
                    data_flow = self._connect_branches(data_flow, 
                        branch_data[_key]['data_flow'],
                        branch_data[_key]['decrease_scale'],
                        branch_data[_key]['connection'])

                data_flow = tf.verify_tensor_all_finite(data_flow, 
                    "output of {}-{} is not finite".format(
                    self.name_scope, layer.name_scope))

                # another task
                if hasattr(self, 'another_task_index'):
                    if self.another_task_index == index:
                        another_output = data_flow
                else: another_output = None
        
        self._logger()
        if another_output is None: return data_flow
        else: return another_output, data_flow