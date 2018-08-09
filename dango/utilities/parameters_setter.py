# -*- coding: utf-8 -*-

'''Set default arguments.'''

import argparse


# arguments types
def _tuple(dtype):
    '''Convert `input-str` into a tuple whose elements are all 
    integers, or floats, or strings.

    NOTE: each elements of 'input-str' must be split by comma now.
    '''
    def wrapper(fun):
        def inner(input_str):
            assert isinstance(input_str, str)
            _tuple = input_str.split(',')
            return tuple([dtype(item.strip()) for item in _tuple])
        return inner
    return wrapper


@_tuple(str)
def str_tuple(input_str):
    return


@_tuple(int)
def int_tuple(input_str):
    return


@_tuple(float)
def float_tuple(input_str):
    return


def str2tuple(input_str):
    return tuple(eval(input_str))


def str2bool(input_str):
    input_str = eval(input_str)
    assert isinstance(input_str, bool)
    return input_str


def int_tuple_list(input_str):
    input_str = eval(input_str)
    assert isinstance(input_str, (int, list, tuple))
    return input_str


# set arguments
def set_system_arguments(parser):
    '''Set default arguments for overall system.'''
    # environment
    parser.add_argument('--task', type=str_tuple,
        help="network's task(s).")

    parser.add_argument('--network', type=str, 
        help='network name.')

    parser.add_argument('--data', type=str,
        help='path to the dataset or tfrecord files.')

    parser.add_argument('--model', type=str,
        help="path to save or restore trained models.")

    parser.add_argument('--action', type=str,
        choices=['train', 'infer'], default='train',
        help='using network to train or infer(default is train).')

    # network
    parser.add_argument('--input_size', type=int_tuple_list, default=224,
        help="input size of the network, can be 'int', 'list' or 'tuple'.")

    parser.add_argument('--input_channel', type=int, default=1,
        help='input channel of the network, default is 1.')

    parser.add_argument('--num_label', type=int_tuple,
        help='number of labels for each task(s).')

    parser.add_argument('--batch', type=int, default=32,
        help='batch size of input data to the network.')

    parser.add_argument('--loss', type=str_tuple, default='cross_entropy', 
        help='loss(es) for task(s).')

    # data
    parser.add_argument('--num_instance', type=int, default=None,
        help='number of instances to be used(for each label).')

    parser.add_argument('--using_records', type=str2bool, default=False,
        help='whether using records to train networks or not.')

    parser.add_argument('--seglabels', type=int_tuple, 
        default=None, help='labels for segmentation.')

    parser.add_argument('--num_crxval', type=int, default=5,
        help='number of folds in cross validation.')

    parser.add_argument('--crxval_index', type=int, default=0,
        help='index of validation folder(start from 0).')
    
    parser.add_argument('--val_fraction', type= float, 
        default=0.1, help='fraction for validation.')

    # image manipulation
    parser.add_argument('--sizes', type=str2tuple, default=[224],
        help="sizes for resizing images, input should be in "
            "a list or tuple form.")
    
    parser.add_argument('--side', type=str, default=None,
        help="one of None, 'shorter' and 'longer'.")
    
    parser.add_argument('--with_crop', type=str2bool, default=False,
        help='whether crop images.')

    parser.add_argument('--crop_num', type=int, default=6,
        help='number to crop along each side.')
    
    parser.add_argument('--with_pad', type=str2bool, default=False,
        help='whether using padding.')

    parser.add_argument('--min_pad', type=int, default=40,
        help='minimal paddings.')
    
    parser.add_argument('--with_rotation', type=str2bool, default=False,
        help='whether using rotation.')

    parser.add_argument('--rotation_range', type=float, default=15,
        help='max angle for rotation.')

    parser.add_argument('--with_histogram_equalisation', type=str2bool, 
        default=False, help='histogram equalisation.')

    parser.add_argument('--with_zero_centralisation', type=str2bool, 
        default=False, help='zeros centralisation.')

    parser.add_argument('--with_normalisation', type=str2bool, 
        default=False, help='normalisation.')

    return parser


def set_train_arguments(parser):
    '''Set arguments for training.'''

    # queue
    parser.add_argument('--capacity', type=int, default=1000,
        help='capacity of input queue.')

    parser.add_argument('--min_after_dequeue', type=int, default=750,
        help='minimal number of elements after dequeue.')

    parser.add_argument('--threads', type=int, default=3,
        help='number of threads to enqueue data.')

    # weight-decay
    parser.add_argument('--weight_decay', type=float, default=1e-5,
        help='weight dacay rate in training.')

    parser.add_argument('--regularizer', type=str,
        choices=['L1', 'L2'], default='L2',
        help='regularizer in training.')

    # optimizer
    parser.add_argument('--simplex', type=str2bool, default=True,
        help='whether using one single optimizer to deal with multitasks.')

    parser.add_argument('--optimizer', type=str_tuple, default='adam',
        help='optimizer(s) for task(s).')

    parser.add_argument('--lr', type=float_tuple, default=1e-5,
        help='initial learning rate for task(s).')

    parser.add_argument('--decay_for', type=str_tuple, default=None,
        help='task(s) for decaying learning rate.')

    parser.add_argument('--decay_type', type=str_tuple, default=None,
        help='scheme for decaying learning rate.')

    parser.add_argument('--decay_step', type=int_tuple, default=1000,
        help='step for decaying learning rate.')

    parser.add_argument('--decay_rate', type=float_tuple, default=0.96,
        help='rate for decaying learning rate.')

    # training loops
    parser.add_argument('--train_iterations', type=int_tuple, 
        default=(0, 10001), help='range of iteration in training.')

    # model saving
    parser.add_argument('--save_every_n', type=int, default=1000,
        help='model saving frequency.')

    parser.add_argument('--checkpoints', type=int, default=100,
        help='maximum number of checkpoints that will be saved.')

    # tensorboard
    parser.add_argument('--tensorboard_every_n', type=int, default=10,
        help='tensorboard summary frequency.')

    # validation
    parser.add_argument('--validate_every_n', type=int, default=30,
        help='frequency of validation or messages to print.')

    return parser


def set_infer_arguments(parser):
    '''Set arguments for inference.'''

    parser.add_argument('--outputs', type=str,
        help='path to save inference outputs.')

    parser.add_argument('--pickles', type=str,
        help='path to load pickel files.')

    parser.add_argument('--infer_iteration', type=int,
        help='using the model in this iteration to do inference.')

    return parser


SUPPORTED_SECTIONS = {
    'SYSTEM': set_system_arguments,
    'TRAIN': set_train_arguments,
    'INFER': set_infer_arguments
}


def set_arguments(parser, section):
    '''Set default values according to `section`.

    param: section: str
        one of SUPPORTED_SECTIONS.keys().
    '''
    if section not in SUPPORTED_SECTIONS.keys():
        raise ValueError('unrecognized section: {}'.format(section))
    return SUPPORTED_SECTIONS[section](parser)


def get_dests(section):
    '''Get `dests` from a parser based on `section`.

    param: section: str
        one of SUPPORTED_SECTIONS.keys().
    return: keywords: list
        dest(s) excluding 'help'.
    '''
    if section not in SUPPORTED_SECTIONS.keys():
        raise ValueError('unrecognized section: {}'.format(section))

    parser = argparse.ArgumentParser()
    parser = SUPPORTED_SECTIONS[section](parser)
    return [action.dest for action in parser._actions
        if action.dest != 'help']
