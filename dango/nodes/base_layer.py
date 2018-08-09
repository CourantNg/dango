# -*- coding: utf-8 -*-

'''BaseLayer and TrainableLayer are two basic layers in layer 
definition, and each layer in network will inherit from one of 
these two basic layers.'''

from dango.nodes.utilities import INITIALIZERS
import tensorflow as tf


class BaseLayer(object):
    '''The most basic layer definition, and `TrainableLayer` 
    also inherits from it.

    `variable_scope` is an important attribute of some layers, 
    and it will be `container_name-layer_name_scope` when each 
    layer instance is called to deal with data flow. 
    
    If layer instance is not under the network directly, then 
    `variable_scope` will be assigned in different way. This 
    can be achieved by giving `variable_scope` as an input in 
    some layers' initialisation. These `variable_scope` can be 
    a str, a layer instance or something else, and will be 
    received by `_variable_scope` as a condidate value of the 
    real `variable_scope`. Finally, when calling layer instances 
    to deal with data flow, the value of `_variable_scope` will 
    be converted to a str to the real desired `variable_scope`.
    
    attribute: name_scope: 
        each layer has its own name scope for aggerating all
        operations in this layer node to a single "big" node
        in `tensorboard`, which is convinent for visualizing
        network structure.
    attribute: variable_scope: 
        scope for initialising variables.
    '''

    def __init__(self, name_scope=None, **kwargs):
        '''
        param: name_scope: str, optional
            name_scope of layer
        '''
        self.name_scope = name_scope
        self.variable_scope = None

    @staticmethod
    def _add(data1, data2):
        '''Add two tensors whose channels may be different, but 
        rest dimensions must be matched to each other.

        If the channel of data1, denoted by C, is smaller than 
        data2's, then data2's `first` C channels will be added 
        by data1, and the rest will be concatenated by the added 
        result and obtain the final result.  

        param: data1, data2: tensorflow.Tensor
            tensor to be added.
        return: data: tensorflow.Tensor
            added tensor. 
        '''
        shape1 = data1.shape.as_list()
        shape2 = data2.shape.as_list()
        assert shape1[:-1] == shape2[:-1]

        if shape1[-1] == shape2[-1]: return data1 + data2

        start_for_add = [0] * len(shape1)
        start_for_concat = [0] * (len(shape1) - 1)
        if shape1[-1] > shape2[-1]:
            data = data2
            data_for_add = tf.slice(data1, start_for_add, shape2)
            data_for_concat = tf.slice(data1, 
                start_for_concat + [shape2[-1]],
                shape1[:-1] + [shape1[-1] - shape2[-1]])
        if shape1[-1] < shape2[-1]:
            data = data1
            data_for_add = tf.slice(data2, start_for_add, shape1)
            data_for_concat = tf.slice(data2, 
                start_for_concat + [shape1[-1]],
                shape1[:-1] + [shape2[-1] - shape1[-1]])
        return tf.concat([data + data_for_add, data_for_concat], axis=-1)

    @staticmethod
    def attach_summaries(tensor, name):
        '''Attach tensor with summaries for visualizing in 
        tensorboard. These summaries includes: histogram,
        max, min, mean, stddev and sparsity.

        param: tensor: tf.Tensor or tf.variable
            tensor to be attached with summaries.
        param: name: str 
            name of these summaries.
        '''
        tf.summary.histogram(name, tensor)
        tf.summary.scalar(name + '-max', tf.reduce_max(tensor))
        tf.summary.scalar(name + '-min', tf.reduce_min(tensor))

        mean = tf.reduce_mean(tensor)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar(name + '-mean', mean)
        tf.summary.scalar(name + 'stddev', stddev)

        tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(tensor))


class TrainableLayer(BaseLayer):
    '''TrainableLayer inherits from BaseLayer. The most difference
    is that it contains `variables` to update in training stage.
    
    attribute: trainable:
        suggests whether to update `variables` or not.
    '''

    def __init__(self, trainable=True, **kwargs):
        '''
        param: trainable: bool, optional
            indicating whether to update `variables` or not.
        '''
        super(TrainableLayer, self).__init__(**kwargs)
        self.trainable = trainable

    def variable_initialiser(self, name, 
                                   shape, 
                                   initializer='he-normal',
                                   regularizer=None,
                                   dtype=tf.float32,
                                   **kwargs):
        '''Initialise variable and attach it with summaries.

        param: name: str
            variable name.
        param: shape: list 
            shape of variable.
        param: initializer: str, optional
            initializer type of variable, one of INITIALIZERS.
        param: regularizer: None or regularizer
            indicating whether variable will be regularized.
        param: dtype: tensorflow.DTYPE 
            data type of variable.
        '''
        assert self.variable_scope, "no variable scope."
        assert initializer in INITIALIZERS, "unrecognized initializer."
        initializer = INITIALIZERS[initializer](**kwargs)

        # initialise variable
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            variable = tf.get_variable(name=name, shape=shape, 
                dtype=dtype, initializer=initializer, 
                trainable=self.trainable)
        
        # attach variable with summaries
        self.attach_summaries(variable, name)
        
        # regularizer
        if regularizer:
            loss = regularizer(variable)
            tf.summary.scalar("{}-regularization".format(name), loss)
            tf.add_to_collection('regularization', loss)

        return variable