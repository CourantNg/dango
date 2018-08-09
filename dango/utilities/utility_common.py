# -*- coding: utf-8 -*-

'''Some commom utilities.'''

from dango.engine.default_supports import SUPPORTED_BASE_TASKS
from dango.engine.default_supports import SUPPORTED_TASKS
import tensorflow as tf
import numpy as np
import logging 
import shutil
import time
import sys
import os


LOG_FORMAT = "\033[1;37m%(levelname)s:\033[1;32mdango:"
LOG_FORMAT += "\033[1;34m%(asctime)s:\033[0m %(message)s"


# warpper
def time_duration(fun):
    def wrapper(*args, **kwargs):
        _time = time.time()
        fun(*args, **kwargs)
        print('Cost time ...{}s'.format(time.time() - _time))
    return wrapper


# task utilities
def task_relates_to(task, task_for):
    '''To judge whether or not a `task` relates to `task_for`. e.g. 
    `task` = 'classification-segmentation' and `task_for` = 'segmentation',
    then this `task` is related to the `task_for`, and return True. 

    param: task: str
        one interested task in SUPPORTED_TASKS.
    param: task_for: str
        base task in SUPPORTED_BASE_TASKS.

    return: True if `task` relates to `task_for`; otherwise, 
        return False.

    raise: AssertionError: 
        if `task` not in SUPPORTED_TASKS or `task_for` not in 
        SUPPORTED_BASE_TASKS/
    '''
    assert task in SUPPORTED_TASKS, ("task " 
        "'{}' is not supported".format(task))
    assert task_for in SUPPORTED_BASE_TASKS, ("base task "
        "'{}' is not supported".format(task_for))

    if task_for in task: return True
    return False


# folders utilities
def touch_folder(path):
    '''Create `path` if not exsits, then return its absolute path.

    param: path: str
        input directory path.
    return: the absolute path of the input `path`.
    '''
    if not os.path.exists(path): os.makedirs(path)
    return os.path.abspath(path)


def touch_latest_folder(path):
    '''Touch the latest folder for saving logs. If not exist, create it. 
    Then return the absolute path the logs folder. 
    
    param: path: str
        directory path of logs folder.
    return: the absolute path of the logs folder.
    '''
    if not os.path.exists(path): 
        folder_name = '0'
    else:
        folders = os.listdir(path)
        if len(folders) == 0: folder_name = '0'
        else: 
            indexes = list()
            for index in folders:
                try: indexes.append(int(index))
                except: pass
            if len(indexes) == 0: folder_name = '0'
            else: folder_name = str(max(indexes) + 1)

    return touch_folder(os.path.join(path, folder_name))


def clean_folder(path):
    '''If `path` doesn't exist, create it; otherwise, 
    clean that directory.
    
    param: path: str
        directory path.
    return: its absolute path.
    '''
    if os.path.exists(path): shutil.rmtree(path)
    return touch_folder(path)

# logger
def set_logger(filename=None):
    tf.logging._logger = logging.getLogger('tensorflow')
    tf.logging._logger.handlers = []
    tf.logging.set_verbosity(tf.logging.INFO)

    std_handler = logging.StreamHandler(sys.stdout)
    _format = logging.Formatter(LOG_FORMAT)
    std_handler.setFormatter(_format)
    tf.logging._logger.addHandler(std_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        _format = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(_format)
        tf.logging._logger.addHandler(file_handler)
        

# computation
def softmax(data):
    '''Softmax function.'''
    assert isinstance(data, np.ndarray)
    data = np.exp(data - np.expand_dims(np.max(data, axis=-1), axis=-1))
    data /= np.expand_dims(np.sum(data, axis=-1), axis=-1)
    return data