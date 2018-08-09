# -*- coding: utf-8 -*-

'''Define backward and evaluation graph.'''

from dango.engine.default_supports import load_loss, load_evaluator
from dango.engine.default_supports import SUPPORTED_BASE_TASKS
from dango.engine.default_supports import SUPPORTED_TASKS
from dango.utilities.utility_common import task_relates_to
from dango.nodes.optimizer import Optimizer
import tensorflow as tf


GTS_DTYPES = {
    'classification': tf.float32,
    'segmentation': tf.int32
}


class BaseNet(object):
    '''BaseNet defines backward and evaluation framework. Each
    network should inherit from it and forward procedure will 
    be defined in it.

    attribute: task: str
        suggests which task will be applied.
    attribute: num_label: dict
        a dict containing base task(s) extracted from `task` as
        key(s), and corresponding `num_label` of each specific 
        task as value. 
    attribute: loss: dict
        a dict containing base task(s) extracted from `task` as 
        key(s), and corresponding `num_label` of each specific
        task as value.
    attribute: gts: dict
        a dict containing base task(s) extraced from `task` as
        keys, and corresponding shape as value.
    attribute: training: tf.bool
        indictes whether in training phase for bn and/or dropout
        layer.
    attribute: eval_training: tf.bool
        suggests which performance to be evaluated: training or 
        validating.
    attribute: logits: dict
        a dict containing base task(s) extracted from `task` as 
        key(s), and corresponding network output as value. 
    attribute: evaluation: dict
        a dict containing base task(s) extracted from `task` as 
        key(s), and corresponding network evaluation as value.
    attribute: merged: tensorflow.Tensor
        contains summaries.
    attribute: optimizations: dict
        If `simplex` is True, then optimizations has just one
        key `task` with corresponding optimization operation as 
        value. 
        If `simplex` is False, it is a dict containing base 
        task(s) extracted form `task` as key(s) and corresponding 
        optimization operation as value.
    '''

    def __init__(self, input_size, 
                       input_channel,
                       num_label, 
                       batch=None,
                       loss=None, 
                       task=None,
                       regularizer=None):
        '''
        param: input_size: int, list or tuple
            input size.
        param: input_channel: int
            input channel.
        param: num_label: int, tuple or dict
            number of labels for task(s) in SUPPORTED_BASE_TASKS.
        param: batch: int or None, optional
            batch size, will be provided generally.
        param: loss: str, tuple or dict
            type of data loss.
        param: task: str or tuple, optional
            task-type.
        param: regularizer:
            regularier function.
        '''
        # check `task`
        if not isinstance(task, (str, tuple)):
            raise TypeError("input 'task' should be str or str of tuple.")
        self.task = task if isinstance(task, str) else '-'.join(task)
        assert self.task in SUPPORTED_TASKS, "unrecognized task."

        # attribute `num_label`` and `loss`` are both dicts whose 
        # keys are extracted from `task`.
        self.num_label = BaseNet._convert_to_dict(num_label, task)
        self.loss = BaseNet._convert_to_dict(loss, task)

        # input definition
        with tf.name_scope('inputs'):
            if task_relates_to(self.task, 'segmentation'):
                if batch is None:
                    raise ValueError("'batch' shouldn't be None in "
                        "segmentation related task, otherwise something "
                        "wrong may happen when using deconvolution.")
            self.input_shapes = BaseNet._generate_input_shapes(
                input_size, batch)
        
            self.input_images = tf.placeholder(
                shape=self.input_shapes['images'] + [input_channel],
                dtype=tf.float32, name='input-images')
            tf.summary.image('input-images', self.input_images, 3)
            tf.summary.histogram('input-images', self.input_images)

            self.gts = dict()
            for _task in SUPPORTED_BASE_TASKS:
                if task_relates_to(self.task, _task):
                    shape=self.input_shapes[_task] + [self.num_label[_task]]
                    self.gts[_task] = tf.placeholder(shape=shape,
                        dtype=GTS_DTYPES[_task], 
                        name="{}-labels".format(_task))

            if len(self.gts.keys()) == 0:
                raise IOError("no ground truth has been loaded yet.")

        self.training = tf.placeholder(tf.bool) # for bn and/or dropout
        self.eval_training = tf.placeholder(tf.bool) # for eavluation
        self.regularizer = regularizer
        self.logits = self.forward()

        tf.logging.info('trainabe variables'.center(100, '*'))
        [tf.logging.info(item) for item in tf.trainable_variables()]
        if self.logits.get('segmentation', None) is not None:
            BaseNet._segmentation_show(self.gts['segmentation'], 
                'segmentation-inputs')
            BaseNet._segmentation_show(self.logits['segmentation'],
                'segmentation-outputs')

    def forward(self):
        '''Forward procedure to obtain `logits`.
        
        return: logits: dict
            keys are different base tasks extracted form `task`
            attribute.
        '''
        raise NotImplementedError

    def _evaluate(self, phase):
        '''Evaluate training or validating performance.
        
        param: phase: str
            one of 'train' or 'validate'.
        return: merged:
            merged summary.
        '''
        all_collections = tf.get_collection(tf.GraphKeys.SUMMARIES)
        for task in SUPPORTED_BASE_TASKS:
            if task_relates_to(self.task, task):
                evaluation_name = "{}-{}-evaluation".format(task, phase)
                tf.summary.scalar(evaluation_name,
                    self.evaluation[task],
                    collections=[evaluation_name])
                all_collections += tf.get_collection(evaluation_name)

        return tf.summary.merge(all_collections)

    def evaluate(self):
        '''Evaluate network and yield `evaluation` and `merged`.'''
        self.evaluation = dict()
        with tf.name_scope('evaluation'):
            for task in SUPPORTED_BASE_TASKS:
                if self.logits.get(task, None) is not None:
                    self.evaluation[task] = load_evaluator(
                        self.loss[task])(self.logits[task], 
                        self.gts[task])

            def true_fun(): return self._evaluate('train')
            def false_fun(): return self._evaluate('validate')
            self.merged = tf.cond(self.eval_training, true_fun, false_fun)

    def _backward(self, optimizers, regularterm):
        '''Backward for different task(s) with each 
        corresponding optimizer.'''
        for task in SUPPORTED_BASE_TASKS:
            if self.logits.get(task, None) is not None:
                loss = load_loss(self.loss[task])(
                    self.logits[task], self.gts[task])
                tf.summary.scalar("{}-loss".format(task), loss)    
                
                loss += regularterm
                tf.summary.scalar("{}-total-loss".format(task), loss)
                
                self.optimizations[task] = optimizers[task](loss)

    def _simplex_backward(self, optimizers, regularterm):
        '''Backward for different task(s) with one 
        single optimizer.'''
        loss = regularterm
        for task in SUPPORTED_BASE_TASKS:
            if self.logits.get(task, None) is not None:
                _loss = load_loss(self.loss[task])(
                    self.logits[task], self.gts[task])
                tf.summary.scalar('{}-loss'.format(task), _loss)
                loss += _loss
        tf.summary.scalar('total-loss', loss)
        self.optimizations[self.task] = optimizers[self.task](loss)

    def backward(self, optimizers, simplex):
        '''Backward graph definition.

        param: optimizers: dict
            a dict containing base task(s) extracted from `task` 
            as key(s) and corresponding optimizer as value.
        param: simplex: bool
            If True, then all tasks' losses will be optimized 
            simultaneously.
        yield: optimizations.
        '''
        assert isinstance(optimizers, dict)
        if not all([isinstance(_, Optimizer) 
            for _ in optimizers.values()]):
            raise ValueError("optimizers haven't been "
                "loaded successfully yet.")
        self.optimizations = dict()

        with tf.name_scope('optimization'):
            try: 
                regularterm = tf.add_n(tf.get_collection('regularization'))
            except: 
                regularterm = 0
            tf.summary.scalar('total-regularization', regularterm)

            if simplex: self._simplex_backward(optimizers, regularterm)
            else: self._backward(optimizers, regularterm)

    @classmethod
    def _convert_to_dict(cls, wanted, task):
        '''It will convert `wanted`, i.e. `num_label` or 
        `loss` into a dict based on `task`.

        param: wanted: int, str, tuple or dict
        param: task: tuple
        '''
        assert isinstance(task, (str, tuple))
        if isinstance(task, str): task = (task, )
        if isinstance(wanted, dict): 
            assert all([key in task for key in wanted.keys()])
            return wanted
        
        returned = dict()
        if isinstance(wanted, (int, str)):
            assert len(task) == 1
            returned[task[0]] = wanted
        else:
            assert isinstance(wanted, tuple)
            assert len(wanted) == len(task)
            for i, _task in enumerate(task):
                returned[_task] = wanted[i]
        return returned

    @classmethod
    def _generate_input_shapes(cls, input_size, batch):
        '''Generate input shapes for different tasks.'''
        assert isinstance(input_size, (int, list, tuple))
        if isinstance(input_size, int): 
            input_size = [batch] + [input_size] * 2
        else: 
            input_size = [batch] + list(input_size)

        input_shapes = {
            'images': input_size,
            'classification': [batch],
            'segmentation': input_size
        }

        return input_shapes

    @classmethod
    def _segmentation_show(cls, feature_maps, name):
        '''Show segmentation inputs and outputs.
        
        param: feature_maps: tensorflow.Tensor
            4-dim tensors.
        param: name: str
            name for `tf.summary.image`.
        '''
        feature_maps = tf.cast(feature_maps, tf.float32) 
        feature_map = feature_maps[0]
        feature_map = tf.transpose(feature_map, perm=[2,0,1])
        shape = feature_map.shape.as_list()
        feature_map = tf.reshape(feature_map, 
            (shape[0], shape[1], shape[2], 1))
        tf.summary.image(name, feature_map, shape[0])