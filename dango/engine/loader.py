# -*- coding: utf-8 -*-

'''Load arguments for `dango.engine.Driver`-instance.'''

from dango.engine.default_supports import SUPPORTED_BASE_TASKS
from dango.engine.default_supports import SUPPORTED_TASKS
from dango.engine.default_supports import load_regularizer
from dango.engine.default_supports import load_optimizer
from dango.engine.default_supports import load_network
from dango.engine.default_supports import load_lr
from dango.utilities.utility_common import touch_latest_folder
from dango.utilities.utility_common import task_relates_to
from dango.utilities.utility_common import touch_folder
from dango.utilities.utility_common import clean_folder
from dango.dataio.image.image_provider import ImageProvider
from tensorflow.python.client import timeline
from PIL import Image
import tensorflow as tf
import numpy as np
import os


class Loader(object):
    '''Load arguments for `engine.Driver`-instance.

    Trained models and logs will be saved in `model`:
        `model` -- models(to save models)
                |- logs(to save logs)
    '''

    def __init__(self, arguments):
        '''
        param: arguments: dict
            a dict containing all arguments based on config file,
            terminal and default values.
        '''
        _sys = arguments['SYSTEM']
        self.action = _sys['action']
        if self.action == 'train': 
            _train, _infer = arguments['TRAIN'], None
        else:
            _train, _infer = None, arguments['INFER']

        # load arguments
        self._load_system(_sys, _train, _infer)
        self._load_network(_sys, _train)
        self._load_data(_sys, _train, _infer)

        # load training settings
        if self.action == 'train': 
            self._load_optimizer(_sys, _train)
            self.network.backward(self.optimizers, self.simplex)
            self.network.evaluate()

        # inference interpreting
        self._interpret_outputs = {
            'classification': 
                self._interpret_classification,
            'segmentation': 
                self._interpret_segmentation,
            'classification-segmentation': 
                self._interpret_classification_segmentation
        }

    def _load_system(self, _sys, _train, _infer):
        '''Load system environment.'''
        # task loading
        self.task = '-'.join(_sys['task'])
        if self.task not in SUPPORTED_TASKS:
            raise ValueError("unrecognized task.")
        tf.logging.info("This application is for "
            "task [{}].".format(self.task))

        # gpu loading
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        # general arguments
        if self.action == 'train':
            self.crxval_index = _sys['crxval_index']
            
            self.logs_savedir = touch_latest_folder(
                os.path.join(_sys['model'], 'logs'))
            self.timeline_savedir = touch_folder(
                os.path.join(_sys['model'], 'timelines'))
            
            self.from_iteration = _train['train_iterations'][0]
            self.to_iteration = _train['train_iterations'][1]
            self.tensorboard_every_n = _train['tensorboard_every_n']
            self.validation_every_n = _train['validate_every_n']
            self.save_every_n = _train['save_every_n']
            self.max_checkpoints = _train['checkpoints']

        if self.action == 'infer':
            self.inference_output = clean_folder(_infer['outputs'])
            self.from_iteration = _infer['infer_iteration']

        self.input_size = _sys['input_size']
        self.input_channel = _sys['input_channel']
        self.batch = _sys['batch'] if self.action == 'train' else 1
        self.model_savedir = touch_folder(
            os.path.join(_sys['model'], 'models'))
        self.restore_path = os.path.join(self.model_savedir,
            'model.ckpt-{}'.format(self.from_iteration))

    def _load_network(self, _sys, _train):
        '''Load network.'''
        self.regularizer = None
        if self.action == 'train':
            self.weight_decay = _train['weight_decay']
            if self.weight_decay > 0:
                self.regularizer = load_regularizer(
                    _train['regularizer'])(self.weight_decay)

        self.network = load_network(_sys['network'])(
            input_size=self.input_size,
            input_channel=self.input_channel,
            num_label=_sys['num_label'],
            batch=self.batch,
            loss=_sys['loss'],
            task=_sys['task'],
            regularizer=self.regularizer)
        tf.logging.info('Network [{}] has been loaded.'.format(
            _sys['network']))

    def _load_data(self, _sys, _train, _infer):
        '''Load data provider.'''
        if self.action == 'train':
            inference = False
            capacity = _train['capacity']
            threads = _train['threads']
            min_after_dequeue = _train['min_after_dequeue']
            pickles = None
        else:
            inference = True
            capacity = 5
            threads = 1
            min_after_dequeue = 3
            pickles = _infer['pickles']

        self.data_provider = ImageProvider(
            datadir=_sys['data'], 
            task=self.task,
            input_size=self.input_size, 
            input_channel=self.input_channel,
            inference=inference, 
            pickles=pickles,
            num_instance=_sys['num_instance'], 
            seglabels=_sys['seglabels'],
            using_records=_sys['using_records'],
            val_fraction=_sys['val_fraction'],
            num_crxval=_sys['num_crxval'], 
            crxval_index=_sys['crxval_index'],
            sizes=_sys['sizes'], 
            side=_sys['side'],
            with_crop=_sys['with_crop'], 
            crop_num=_sys['crop_num'],
            with_pad=_sys['with_pad'], 
            min_pad=_sys['min_pad'],
            with_rotation=_sys['with_rotation'], 
            rotation_range=_sys['rotation_range'],
            with_histogram_equalisation=_sys['with_histogram_equalisation'],
            with_zero_centralisation=_sys['with_zero_centralisation'],
            with_normalisation=_sys['with_normalisation'],
            capacity=capacity, 
            num_threads=threads, 
            min_after_dequeue=min_after_dequeue)
        tf.logging.info('Data provider has been loaded over.')

    def _load_optimizer(self, _sys, _train):
        '''Load optimizer.'''
        self.optimizers = dict()
        self.simplex = _train['simplex']

        if self.simplex:
            optimizer = _train['optimizer'][0]
            init_lr = _train['lr'][0]
            
            if _train['decay_for'] is None:
                lr = load_lr('naive')(init_lr, name=self.task)
            else:
                lr = load_lr(_train['decay_type'][0])(init_lr, 
                    _train['decay_step'][0], _train['decay_rate'][0],
                    name=self.task)
            self.optimizers[self.task] = load_optimizer(optimizer)(lr)
            msg = "Optimizer for [{}] has been loaded".format(self.task)
            tf.logging.info(msg)
        else:
            optimizers = Loader._convert_to_dict(
                _train['optimizer'], _sys['task'])
            init_lrs = Loader._convert_to_dict(
                _train['lr'], _sys['task'])
            
            decay_for = dict()
            for task in _sys['task']:
                decay_for[task] = False
                if _train['decay_for'] is None: continue
                if task in _train['decay_for']: decay_for[task] = True
            if _train['decay_for']:
                decay_types = Loader._convert_to_dict(
                    _train['decay_type'], _train['decay_for'])
                decay_steps = Loader._convert_to_dict(
                    _train['decay_step'], _train['decay_for'])
                decay_rates = Loader._convert_to_dict(
                    _train['decay_rate'], _train['decay_for'])

            for task in SUPPORTED_BASE_TASKS:
                if task_relates_to(self.task, task):
                    if decay_for[task]:
                        lr = load_lr(decay_types[task])(
                            init_lrs[task], decay_steps[task],
                            decay_rates[task], name=task)
                    else: lr = load_lr('naive')(init_lrs[task], name=task)
                    self.optimizers[task] = load_optimizer(
                        optimizers[task])(lr)
                    msg = "Optimizer for [{}] has been loaded.".format(task)
                    tf.logging.info(msg)

    def _load_saver(self, key=None):
        '''Load saver in `Driver._train_network`.

        param: key: str or None
            keyword for fetching variables to restore in
            `Loader._fetch_variables_to_restore`.
        return: saver: tf.train.Saver
            saver.
        return: variable_to_initialise: list
            containing variables to initialise.
        '''
        if key is not None:
            variables_to_restore, variables_to_initialise = \
                Loader._fetch_variables_to_restore(key)
        else:
            variables_to_restore = None
            variables_to_initialise = None

        if self.action == 'infer':
            self.max_checkpoints = 5
        saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints,
            save_relative_paths=True,
            var_list=variables_to_restore)

        return saver, variables_to_initialise

    def _record_timeline(self, run_metadata, iteration):
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        filepath = os.path.join(self.timeline_savedir, 
            'timeline_step_%5d' % iteration)
        with open(filepath, 'w') as f: f.write(chrome_trace)

    def _make_checkpoint(self, saver, sess, iteration):
        '''Save model.'''
        if iteration != self.from_iteration:
            save_path = os.path.join(self.model_savedir, 'model.ckpt')
            saver.save(sess, save_path, iteration)

    def _init_network(self, sess, saver, variables_to_initialise=None):
        '''Initial or restore the network.

        param: variables_to_initialise: list 
            containing variables to initialise.
        '''
        if self.from_iteration == 0:
            sess.run(tf.global_variables_initializer())
        else:
            try:
                saver.restore(sess, self.restore_path)
                if variables_to_initialise is not None:
                    sess.run(tf.variables_initializer(
                        variables_to_initialise))
            except:
                tf.logging.error('{} does not exist'.format(
                    self.restore_path))
                exit()

    def _interpret_classification(self, logits, sth, save=True):
        logits = np.argmax(logits['classification'], axis=1)
        logits = np.argmax(np.bincount(logits))
        label = self.data_provider.label_codes['classification'][logits]
        sth['name'] = label + '_' + sth['name']
        if save:
            image = Image.open(sth['path'])
            image.save(os.path.join(self.inference_output, sth['name']))
        return sth

    def _interpret_segmentation(self, logits, sth):
        for i in range(logits['segmentation'].shape[0]):
            logit = np.argmax(logits['segmentation'][i], axis=2)
            image = np.zeros_like(logit, dtype=np.uint8)
            for k in range(logit.shape[0]):
                for j in range(logit.shape[1]):
                    image[k][j] = self.data_provider.label_codes[
                        'segmentation'][logit[k][j]]
            image = Image.fromarray(image)
            image.save(os.path.join(self.inference_output, 
                str(i) + '_' + sth['name']))
        
    def _interpret_classification_segmentation(self, logits, sth):
        sth = self._interpret_classification(logits, sth, save=False)
        self._interpret_segmentation(logits, sth)

    @classmethod
    def _convert_to_dict(cls, option, based_on_option):
        '''Convert `option` into a dict based on `based_on_option`.
        
        param: option: tuple
            option will be converted into a dict.
        param: based_on_option: tuple
            reference tuple.
        return: returned: dict
            returned dict.
        '''
        assert isinstance(option, tuple)
        assert isinstance(based_on_option, tuple)

        num = len(based_on_option)
        if len(option) == 1: option *= num
        if len(option) != num:
            raise ValueError("{} mismatches {}".format(
                str(option), str(based_on_option)))

        returned = dict()
        for index, item in enumerate(based_on_option):
            returned[item] = option[index]
        return returned

    @classmethod
    def _summaries_writer(cls, phase, savedir):
        '''Generate summaries writer.'''
        assert phase in ['train', 'validate']
        logs =  touch_folder(os.path.join(savedir, phase))
        writer = tf.summary.FileWriter(logs)
        return writer

    @classmethod
    def _fetch_variables_to_restore(cls, key):
        '''Fetch variables to restore.

        param: key: str
            indicating which variable will be restored.
        return: variables_to_restore: list
            containing variables to restore.
        return: variables_to_initialise: list
            containing variables to initialise.
        '''
        excluding = ['Adam', 'Momentum']
        def is_excluding(name):
            '''If returen True, then `name` excludes `excluding`.'''
            return not any([i in name for i in excluding])

        variables_to_restore = list()
        variables_to_initialise = list()
        for variable in tf.global_variables():
            if is_excluding(variable.name) and key in variable.name:
                variables_to_restore.append(variable)
            else:
                variables_to_initialise.append(variable)
        return variables_to_restore, variables_to_initialise
