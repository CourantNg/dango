# -*- coding: utf-8 -*-

import importlib


SUPPORTED_BASE_TASKS = [
    'classification', 
    'segmentation'
]


SUPPORTED_TASKS = [
    'classification',
    'segmentation',
    'classification-segmentation'
]


SUPPORTED_NETWORKS = {
    'testnet':
        'dango.networks.TestNet.TestNet'
}


SUPPORTED_DECAY_TYPES = {
    'naive':
        'dango.nodes.learning_rate.LearningRate',
    'exponential':
        'dango.nodes.learning_rate.ExponentialDecayedLearningRate'
}


SUPPORTED_OPTIMIZERS = {
    'adam':
        'dango.nodes.optimizer.AdamOptimizer',
    'momentum':
        'dango.nodes.optimizer.MomentumOptimizer',
    'gradient':
        'dango.nodes.optimizer.GradientOptimizer'
}


SUPPORTED_LOSSES = {
    'cross_entropy':
        'dango.nodes.data_loss.cross_entropy_loss',
    'hinge':
        'dango.nodes.data_loss.hinge_loss',
    'dice':
        'dango.nodes.data_loss.dice_loss',
    'dice_hinge':
        'dango.nodes.data_loss.dice_hinge_loss',
    'weighted_dice':
        'dango.nodes.data_loss.weighted_dice_loss',
}


SUPPORTED_EVALUATORS = {
    'cross_entropy':
        'dango.nodes.evaluate.accuracy_evaluator',
    'dice':
        'dango.nodes.evaluate.hard_dice_evaluator',
    'dice_hinge':
        'dango.nodes.evaluate.hard_dice_evaluator',
    'weighted_dice':
        'dango.nodes.evaluate.hard_dice_evaluator',
}


SUPPORTED_REGULARIZERS = {
    'l1':
        'dango.nodes.regularizer.l1_regularizer',
    'l2':
        'dango.nodes.regularizer.l2_regularizer'
}


def _load(name, supported):
    '''Load class according to provided `name` in `supported`.

    param: name: str
        to locate the corresponding class to load.
    param: supported: dict
        where the class can be founded.
    
    return: _class: cls
        class corresponding to provided `name`.
    '''
    name = name.lower()
    assert name in supported.keys(), 'unrecognized {}'.format(name)
    module_name, class_name = supported[name].rsplit('.', 1)
    module = importlib.import_module(module_name)
    _class = getattr(module, class_name)
    return _class


def load_network(network):
    return _load(network, SUPPORTED_NETWORKS)


def load_lr(decay):
    return _load(decay, SUPPORTED_DECAY_TYPES)


def load_optimizer(optimizer):
    return _load(optimizer, SUPPORTED_OPTIMIZERS)


def load_loss(loss):
    return _load(loss, SUPPORTED_LOSSES)


def load_evaluator(loss):
    return _load(loss, SUPPORTED_EVALUATORS)


def load_regularizer(reg):
    return _load(reg, SUPPORTED_REGULARIZERS)