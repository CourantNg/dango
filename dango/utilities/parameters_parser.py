# -*- coding: utf-8 -*-

'''Parse user-defined config file.'''

from dango.utilities.parameters_setter import set_arguments, get_dests
from dango.utilities.parameters_setter import SUPPORTED_SECTIONS
from dango.utilities.utility_common import touch_folder, set_logger
import tensorflow as tf
import configparser
import argparse
import datetime
import os


SECTIONS = list(SUPPORTED_SECTIONS.keys())


def parse():
    '''`parser` is firstly used to locate the user defined config 
    file according to command in the terminal. Then, config file 
    will be parsed to load arguments.

    return: arguments: dict
        a group of parameters defined in config file and/or terminal.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='location of config file.')
    config_path, arguments_from_cmd = parser.parse_known_args()

    # check config_path
    assert config_path.conf, 'no config file has been provided yet.'
    if not os.path.isfile(config_path.conf):
        raise IOError('no config file: {}.'.format(config_path.conf))
    config_path = os.path.abspath(config_path.conf)

    # parse config file
    config = configparser.ConfigParser()
    config.read(config_path)
    default_keywords = _check_keywords(config)

    arguments = dict()
    for section in config.sections():
        arguments[section] = dict()
        section_arguments, arguments_from_cmd = \
            _overrides(section, config, arguments_from_cmd)
        for option in default_keywords[section]:
            arguments[section][option] = \
                getattr(section_arguments, option)

    assert not arguments_from_cmd, "unrecognized argument(s) from terminal."
    _check_arguments(arguments)
    _set_logger(arguments, config_path)
    
    return arguments


def _check_keywords(config):
    '''Check `sections` and `options` in config file against 
    arguments defined in `dango.utilities.parameters_setter`.'''
    # check sections
    if not all([sec.upper() in SECTIONS for sec in config.sections()]):
        msg = "unrecognized section in {}, which should be in {}".format(
            str(config.sections()), str(SECTIONS))
        raise ValueError(msg)

    # check options
    default_keywords = dict()
    for section in config.sections():
        default_keywords[section] = get_dests(section)
        items = config.items(section)
        if items:
            config_keywords = list(dict(items))
            for option in config_keywords:
                if option in default_keywords[section]: 
                    continue
                else: 
                    raise ValueError("unrecognized option "
                    "{} in section {}".format(option, section))
    
    return default_keywords


def _overrides(section, config, arguments_from_cmd):
    '''Overrides section arguments' default values from a config 
    file and terminal.'''
    parser = argparse.ArgumentParser()
    parser = set_arguments(parser, section)

    arguments_from_config = []
    for key, value in dict(config.items(section)).items():
        arguments_from_config.extend(['--{}'.format(key), str(value)])

    arguments, arguments_from_cmd = parser.parse_known_args(
        arguments_from_config + arguments_from_cmd)

    return arguments, arguments_from_cmd


def _check_arguments(arguments):
    '''Assert some arguments' values.

    param: arguments: dict
        all arguments from config file and terminal.
    '''
    # check `task`, 'network'
    if not arguments['SYSTEM']['task']:
        raise ValueError("not found 'task' in config file or terminal.")
    if not arguments['SYSTEM']['network']:
        raise ValueError("not found 'network' in config file or terminal.")
    
    # check `data``
    data = arguments['SYSTEM']['data']
    try: data = touch_folder(data)
    except: raise ValueError("'{}' is not a reasonable path.".format(data))
    if len(os.listdir(data)) == 0:
        raise IOError("not found data in '{}'.".format(data))
    arguments['SYSTEM']['data'] = data

    # check `action`, `model`, `outputs`, `pickles`
    model = arguments['SYSTEM']['model']
    if arguments['SYSTEM']['action'] == 'train':
        assert arguments['TRAIN']
        arguments['SYSTEM']['model'] = touch_folder(model)
    if arguments['SYSTEM']['action'] == 'infer':
        assert arguments['INFER']
        if not os.path.isdir(model) or len(os.listdir(model)) == 0:
            raise IOError("not found trained models in '{}'.".format(model))
        try:
            arguments['INFER']['outputs'] = touch_folder(
                arguments['INFER']['outputs'])
        except:
            raise ValueError("'{}' is not a reasonable path.".format(
                arguments['INFER']['outputs']))
        try:
            arguments['INFER']['pickles'] = touch_folder(
                arguments['INFER']['pickles'])
            if len(os.listdir(arguments['INFER']['pickles'])) == 0:
                raise IOError("not found pickle files in '{}'.".format(
                    arguments['INFER']['pickles']))
        except:
            raise ValueError("'{}' is not a reasonable path.".format(
                arguments['INFER']['pickles']))
    
    # check `capacity`, `min_after_dequeue`, `batch`
    if arguments.get('TRAIN', None) is not None:
        arguments['TRAIN']['capacity'] = max(arguments['TRAIN']['capacity'],
            arguments['TRAIN']['min_after_dequeue'] + \
            3 * arguments['SYSTEM']['batch'])


def _write_arguments(arguments, path):
    '''Write `arguments` to text file for future reference.'''
    _file = '{}-arguments.txt'.format(arguments['SYSTEM']['action'])
    _path = os.path.join(path, _file)

    outputs = ['Parameters defined at {}'.format(
        str(datetime.datetime.now())[:-6])]

    for section in arguments.keys():
        print('[{}]'.format(section))
        outputs.append('[{}]'.format(section))
        for option in arguments[section].keys():
            option = '-- {}: {}'.format(
                option, arguments[section][option])
            print(option)
            outputs.append(option)

    with open(_path, 'a+') as f:
        [f.write(line + os.linesep) for line in outputs]


def _set_logger(arguments, config_path):
    '''Logger.'''
    _action = arguments['SYSTEM']['action']
    logs = '{}-logs'.format(_action)

    _path = arguments['SYSTEM']['model']
    if _action == 'infer':
        _path = os.path.dirname(os.path.abspath(_path))
    logs = os.path.join(_path, logs)
    
    set_logger(logs)
    _write_arguments(arguments, _path) 
    
    tf.logging.info('Parse arguments over in {}'.format(config_path))