# -*- coding: utf-8 -*-

'''Provider image data and corresponding labels and/or masks
(mask image for segmentation is always in GRAY format).'''

from dango.dataio.image.records_generator import RecordsGenerator
from dango.dataio.image.image_producer import ImageProducer
from dango.engine.default_supports import SUPPORTED_BASE_TASKS
from dango.utilities.utility_common import task_relates_to
from dango.dataio.image.directory_parser import isimage
from collections import OrderedDict
from random import uniform, randint
from pickle import load
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

class ImageProvider(ImageProducer):
    '''Provide next batch data to networks.
    
    attribute: feature_keys: list
        containing keys for decoding records.
    attribute: one_hotting_codes: dict
        holding one hotting codes for related base task(s) 
        in training.
    attribute: label_codes: dict
        holding label codes for related base task(s) in 
        inference.
    attribute: records: dict in training and list in inference
        1) in training stage, key 'train' and/or 'validate' exist. 
        if `using_records` is True, then each element in 
        'train' and/or 'validate' is the path of tfrecord file. 
        if `using_records` is False, then each element in 
        'train' and/or 'validate' is the path of csv file.
        2) in inference stage, each element is the image path.
    '''

    def __init__(self, datadir, 
                       task,
                       input_size=224,
                       input_channel=1, 
                       inference=False, 
                       pickles=None,
                       num_instance=None,
                       seglabels=None, 
                       using_records=False,
                       val_fraction=0.1,
                       num_crxval=5, 
                       crxval_index=0,
                       sizes=[224, 256], 
                       side=None,
                       with_crop=False, 
                       crop_num=6,
                       with_pad=True, 
                       min_pad=40,
                       with_rotation=False, 
                       rotation_range=15,
                       with_histogram_equalisation=False,
                       with_zero_centralisation=False,
                       with_normalisation=False,
                       capacity=1000, 
                       min_after_dequeue=750,
                       num_threads=3):
        '''Refer to doc of `ImageProducer` defined in 
        `dango.dataio.image.image_producer`.

        param: datadir: str
            directory of image dataset. 
            in inference stage, this directory may be not in `datalibs`. 
        param: pickles: str or None, optional
            path of pickle files(should be provided in inference stage).
        param: crxval_index: int or None, optional
            indicating which tfrecord file is used for validation.
            if None, then no validation.
        param: with_rotation: bool
            indicating whether or not images will be rotated.
        param: rotation_range: float
            range of rotation: [-rotation_range, rotation_range].
        param: with_histogram_equalisation: bool
            indicating whether using histogram equalisation.
        param: with_zero_centralisation: bool
            indicating whether using zero centralisation.
        param: with_normalisation: bool
            indicating whether using normalisation.
        param: capacity: int
            capacity of input queue.
        param: min_after_dequeue: int
            minimal examples remained after dequeueing.
        param: num_threads: int
            number of threads for queueing input queue.
        '''
        super(ImageProvider, self).__init__(
            datadir=datadir, 
            task=task,
            input_size=input_size,
            input_channel=input_channel,
            inference=inference, 
            num_instance=num_instance, 
            seglabels=seglabels, 
            using_records=using_records, 
            num_crxval=num_crxval,
            val_fraction=val_fraction,
            sizes=sizes, 
            side=side,
            with_crop=with_crop, 
            crop_num=crop_num,
            with_pad=with_pad, 
            min_pad=min_pad
        )

        if task_relates_to(self.task, 'segmentation'):
            self.num_seglabels = len(self.seglabels)

        self.with_rotation = with_rotation
        self.rotation_range = rotation_range
        self.with_histogram_equalisation = with_histogram_equalisation
        self.with_zero_centralisation = with_zero_centralisation
        self.with_normalisation = with_normalisation

        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.num_threads = num_threads

        self._load_pickles(pickles)
        self._load_records(crxval_index)
        
    def _load_pickles(self, pickles):
        '''In training stage, whatever `datadir` or `tfrecordir`
        attribute exists, the path for storing pickle files can 
        always be found by calling `ImageProducer.look_up_path`.

        In inference stage, `datadir` may be changed. For instance, 
        in training stage, `datadir` is `app` in `datalibs` then 
        pickle files are in `datarecords.app.pickles`; however, in 
        inference stage, `datadir` is `app-infer`, then `pickles` should 
        be provided to find pickle files in `datarecords.app.pickles`.

        param: pickles: str or None
            path for storing pickle files.
        '''
        self.feature_keys = ['image']
        pickles = pickles if self.inference else self._look_up_path()
        self.one_hotting_codes, self.label_codes = dict(), dict()
        for task in SUPPORTED_BASE_TASKS:
            if task_relates_to(self.task, task):
                _pickle = os.path.join(pickles, '{}.pk'.format(task))
                if not os.path.exists(_pickle):
                    raise IOError("{}.pk doesn't be provided.".format(task))
                with open(_pickle, 'rb') as fp:
                    _loaded = load(fp)
                    self.one_hotting_codes[task] = \
                        _loaded["{}-one-hotting".format(task)]
                    self.label_codes[task] = _loaded['{}-label'.format(task)]
                self.feature_keys.append(task)
    
    def _get_records_by_tfrecords(self, crxval_index):
        '''Get records by tfrecord files. This function is called 
        only when in training stage and `using_recrods` is True.
        
        If `tfrecordir` attribute exists, the existence of tfrecord 
        files in `datarecords.app.records` has already been checked 
        in `dango.dataio.image_producer._check`.

        If `datadir` attribute exists, the existence of tfrecord files
        in `datarecords.app.records` should be checked, and will be 
        generated if not exists.

        param: crxval_index: int
            indicating which tfrecord file is used for validation.
            if None, then no validation.
        '''
        if hasattr(self, 'datadir'):
            self.tfrecordir = self._look_up_path('records')
            _tfrecords = [_file for _file in 
                os.listdir(self.tfrecordir) if '.tfrecords' in _file]
            if len(_tfrecords) == 0:
                RecordsGenerator(
                    datadir=self.datadir, 
                    task=self.task, 
                    input_size=self.input_size, 
                    input_channel=self.input_channel,
                    num_instance=self.num_instance, 
                    seglabels=self.seglabels, 
                    num_crxval=self.num_crxval,
                    sizes=self.sizes, 
                    side=self.side,
                    with_crop=self.with_crop, 
                    crop_num=self.crop_num,
                    with_pad=self.with_pad, 
                    min_pad=self.min_pad)()

        self.records = {'train': list(), 'validate': list()}
        _tfrecords = [_file for _file in 
            os.listdir(self.tfrecordir) if '.tfrecords' in _file]
        for _file in _tfrecords:
            _file_path = os.path.join(self.tfrecordir, _file)
            if crxval_index is None:
                self.records['train'].append(_file_path)
                continue
            if crxval_index == int(re.findall(r'\d+', _file)[0]):
                self.records['validate'].append(_file_path)
            else:
                self.records['train'].append(_file_path)
        if len(self.records['validate']) == 0: self.records.pop('validate')
    
    def _get_records_by_csv(self):
        '''Get records by csv files. This function is called only
        when in training stage and `using_records` is False.
        
        crxvals` obtained by `dataio.image_producer._parse_directory_trees` 
        may don't have key `validate`:
            crxvals = {
                    'train': 
                        {'label-1': [...], 'label-2': [...]}, 
                    'validate': 
                        {'label-1': [...], 'label-2': [...]}
                }
        
        Csv files will be created if not exist, and the "virtual" 
        header of these csv files are `image`, 'classification' 
        and/or `segmentation` based on `task`.'''
        assert hasattr(self, 'datadir')
        crxvals = self._parsing_directory_trees()

        save_path = self._look_up_path()
        csvs = [_file for _file in 
            os.listdir(save_path) if '.csv' in _file]

        self.records = dict()
        if len(csvs) == 0:    
            for state in crxvals.keys():
                _data = OrderedDict()
                for _key in self.feature_keys: 
                    _data[_key] = list()
                for label in crxvals[state].keys():
                    for instance in crxvals[state][label]:
                        if isinstance(instance, (tuple, list)):
                            _data['image'].append(instance[0])
                        if isinstance(instance, str):
                            _data['image'].append(instance)
                        for _key in self.feature_keys:
                            if _key == 'classification':
                                _data[_key].append(label)
                            if _key == 'segmentation':
                                _data[_key].append(instance[1])

                csv_name = '{}.csv'.format(state)
                dataframe = pd.DataFrame(_data)                        
                dataframe.to_csv(os.path.join(save_path, csv_name),
                    header=False, index=False)
                self.records[state] = [os.path.join(save_path, csv_name)]

        if len(csvs) > 0:
            for _csv in csvs:
                csv_path = os.path.join(save_path, _csv)
                if 'train.csv' == _csv:
                    if not os.path.exists(csv_path):
                        raise IOError("no train.csv in "
                            "'{}'.".format(save_path))    
                    self.records['train'] = [csv_path]
                if 'validate.csv' == _csv:
                    self.records['validate'] = [csv_path]

    def _get_records_in_inference(self):
        '''Get records in inference stage.
        
        In inference stage, tfrecord files will not be provided, and
        `records` are a list containing image paths as elements.'''
        assert hasattr(self, 'datadir')
        self.records = [os.path.join(self.datadir, _file) 
            for _file in os.listdir(self.datadir) 
            if isimage(_file)]

    def _load_records(self, crxval_index):
        '''Load `records` according to `using_records` and 
        `inference`. 

        In training stage, if `using_records` is True, then
        `_get_records_by_tfrecords` will be called; otherwise,
        `_get_records_by_csv` will be called if `using_records`
        is False. 

        In inference stage, `_get_records_in_inference` will be 
        called.'''
        if not self.inference:
            if self.using_records:
                self._get_records_by_tfrecords(crxval_index)
            else:
                self._get_records_by_csv()
            for phase in self.records.keys():
                _info = "{}-records".format(phase).center(100, '*')
                tf.logging.info(_info)
                for item in self.records[phase]:
                    tf.logging.info(item)
        else:
            self._get_records_in_inference()

    def _decode_records(self, files):
        '''Decoding records to load images and corresponding 
        labels and/or masks.

        param: files: list
            containing tfrecord files or csv files.
        return: tensors: dict
            {
                'image': image-tensor # must have
                'classification': label-tensor # may have
                'segmentation': mask-tensor # may have
            }
        '''
        with tf.name_scope('input-queue'):
            filequeue = tf.train.string_input_producer(files)
            
            if self.using_records: # using tfrecord files
                _, example = tf.TFRecordReader().read(filequeue)
                _features = dict()
                for _key in self.feature_keys:
                    _features[_key] = tf.FixedLenFeature([], tf.string)
                features = tf.parse_single_example(example, features=_features)
                
                image = tf.decode_raw(features['image'], tf.uint8)
                image = tf.reshape(image, self.input_size+[self.input_channel])
                tensors = {'image': image}
                if features.get('classification', None) is not None: 
                    tensors['classification'] = features['classification']
                if features.get('segmentation', None) is not None:
                    mask = tf.decode_raw(features['segmentation'], tf.uint8)
                    mask = tf.reshape(mask, self.input_size+[1])
                    tensors['segmentation'] = mask
            else: # using csv files
                _, example = tf.TextLineReader().read(filequeue)
                tensors = dict()
                record_defaults = [["None"]] * len(self.feature_keys)
                if self.task == 'classification':
                    tensors['image'], tensors['classification'] = \
                        tf.decode_csv(example, record_defaults=record_defaults)
                if self.task == 'segmentation':
                    tensors['image'], tensors['segmentation'] = \
                        tf.decode_csv(example, record_defaults=record_defaults)
                if self.task == 'classification-segmentation':
                    tensors['image'], tensors['classification'], \
                        tensors['segmentation'] = \
                        tf.decode_csv(example, record_defaults=record_defaults)

        return tensors
        
    def _next_data(self, files, batch):
        '''Load the next batch images and corresponding labels 
        or masks in training.

        param: files: list
            containing tfrecord files or csv files.
        param: batch: int
            batch size.
        return: tensors: dict
            {
                'image': image-tensor # must have
                'classification': label-tensor # may have
                'segmentation': mask-tensor # may have
            }
        '''
        _tensors = self._decode_records(files)
        tensors = tf.train.shuffle_batch(_tensors,
            batch_size=batch, capacity=self.capacity,
            min_after_dequeue=self.min_after_dequeue,
            num_threads=self.num_threads)
        return tensors

    def _next_data_in_inference(self, path):
        '''Load next data in inference.

        param: path: str
            image path.
        return: images: np.ndarray
            input of the network.
        return: a dict
            `name`: output name for saving;
            `path`: output path for saving.
        '''
        name, suffix = os.path.splitext(os.path.basename(path))
        name = name + '_seg' + suffix

        images = list()
        for size in self.sizes: 
            _images = self._get_image(path, size, 
                mode=self.input_channel, inference=True)
            for image, _ in _images:
                image = np.squeeze(np.asarray(image, dtype=np.uint8))
                if self.with_histogram_equalisation:
                    image = self.image_equalize_hist(image)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                images.append(np.expand_dims(image, axis=0))
        
        images = np.concatenate(images) # 4-dim
        if self.with_zero_centralisation:
            images = self.image_zero_centralization(images)
        if self.with_normalisation:
            images = self.image_normalisation(images)
        if not any([self.with_zero_centralisation, self.with_normalisation]):
            images = self.image_rescale(images)

        return images, {'name': name, 'path': path}

    def next_data(self, batch=None):
        '''Provide next tensors to be fed into network.'''
        if not self.inference:
            assert batch, "'batch' should be provided in training."
            tensors = dict()
            for phase in self.records.keys():
                tensors[phase] = self._next_data(self.records[phase], batch)
            return tensors
        else:
            try:
                return self._next_data_in_inference(self.records.pop(0))
            except IndexError:
                return None

    def _one_hotting_mask(self, mask):
        '''Generate one hotting code for mask image in segmentation 
        related task.

        param: mask: numpy.ndarray
            mask image.
        return: one_hotting: np.ndarray
            4-dim one hotting code.
        '''
        mask = np.squeeze(mask)
        assert len(mask.shape) == 2
        shape = mask.shape + (self.num_seglabels,)
        one_hotting = np.zeros(shape, dtype=np.uint8)

        for i in range(shape[0]):
            for j in range(shape[1]):
                try:
                    one_hotting[i][j] = \
                        self.one_hotting_codes['segmentation'][mask[i][j]]
                except:
                    raise ValueError("In segmentation, unexpected " 
                        "label {} emerges.".format(mask[i][j]))
        one_hotting = np.expand_dims(one_hotting, axis=0)
        return one_hotting

    def _get_image(self, image, size, mode=None, 
                         offset=None, inference=False):
        '''Get image and do image manipulations: resizing, padding 
        or cropping.
        
        If `image` is str or bytes, it must come from csv files
        or in inference stage, thus image resizing, cropping or 
        padding should be applied. In addition, in training
        stage, cropping will just be applied at almost once.

        If `image` is numpy.ndarray, then resizing and cropping or
        padding have been applied already.

        param: image: str, bytes or np.ndarray
            input image.
        param: size: int, list or tuple
            desired size to be resized.
        param: mode: int, str or None
            1, 3, 'L', 'RGB' or None, the mode of image.
        param: offset: tuple or None
            starting point for cropping when `once` is True.
        param: inference: bool
            indicating whether or not it's in inference.
        '''
        assert isinstance(image, (np.ndarray, str, bytes))

        if isinstance(image, (str, bytes)): 
            image = self.image_open(image, mode=mode)
            image, _ = self.image_resize(image, size, side=self.side)
            
            if self.with_crop:
                once = False if inference else True
                image, offset = self.image_crop(image, self.input_size, 
                    num=self.crop_num, once=once, offset=offset)
            if self.with_pad:
                image = self.image_pad(image, min_pad=self.min_pad)
            
            if inference: return image
            else: image = image[0][0]

        image = np.squeeze(np.asarray(image, dtype=np.uint8))
        if len(image.shape) == 2: 
            image = np.expand_dims(image, axis=-1)
        
        return image, offset

    def _get_piece(self, batches, piece):
        '''Get a piece of data: 
            image <--> batches['image'][piece],
            label <--> batches['classification'][piece],
            mask  <--> batches['segmentation'][piece]
        then do image manipulations to image and/or mask for image
        augmentation: fliping, rotation, etc.

        param: batches: dict
            {
                'image': numpy.ndarray, # must have
                'classification': numpy.ndarray, # may have
                'segmentation': numpy.ndarray # may have
            }
        param: piece: int
            index of piece-data in the batches['image'].
        return: image: np.ndarray
            image array
        return: label: np.ndarray or None
            one hotting codes or None.
        return: mask: np,ndarray or None
            one hotting codes or None.
        '''
        size = self.sizes[randint(0, len(self.sizes) - 1)]
        image, offset = self._get_image(batches['image'][piece], 
            size, mode=self.input_channel)
        if self.with_histogram_equalisation:
            image = self.image_equalize_hist(image)
        if self.with_rotation:
            rotation_flag = randint(0, 1) 
            rand_angle = uniform(-self.rotation_range, self.rotation_range)
            if rotation_flag: image = self.image_rotate(image, rand_angle)

        if batches.get('classification', None) is not None:
            label = batches['classification'][piece].decode('utf-8')
            label = self.one_hotting_codes['classification'][label] # 2-dim
        else: label = None

        if batches.get('segmentation', None) is not None:
            mask, _ = self._get_image(batches['segmentation'][piece], 
                size, mode=1, offset=offset)
            if self.with_rotation and rotation_flag:
                mask = self.image_rotate(mask, rand_angle, dtype=np.uint8)
        else: mask = None
        
        image, mask = self.image_transform(image, mask)
        image = np.expand_dims(image, axis=0) # 4-dim
        if mask is not None:
            # mask = np.expand_dims(mask, axis=0) # 4-dim
            mask = self._one_hotting_mask(mask) # 4-dim

        return image, label, mask

    def convert_to_array(self, tensors, sess):
        '''Convert tensors to numpy.ndarray to be fed into network.

        param: tensors: dict
            {
                'image': image-tensor, # must have
                'classification': label-tensor, # may have
                'segmentation': mask-tensor, # may have
            }
        return: batches: dict
            {
                'image': numpy.ndarray, # must have
                'classification': numpy.ndarray, # may have
                'segmentation': numpy.ndarray # may have
            }
        '''
        assert not self.inference
        batches = sess.run(tensors)

        images, labels, masks = self._get_piece(batches, piece=0)
        for i in range(1, batches['image'].shape[0]):
            image, label, mask = self._get_piece(batches, piece=i)
            images = np.concatenate((images, image))
            if labels is not None: 
                labels = np.concatenate((labels, label))
            if masks is not None: 
                masks = np.concatenate((masks, mask))
        
        batches['image'] = images
        if labels is not None: 
            batches['classification'] = labels
        if masks is not None: 
            batches['segmentation'] = masks

        if self.with_zero_centralisation:
            batches['image'] = self.image_zero_centralization(batches['image'])
        if self.with_normalisation:
            batches['image'] = self.image_normalisation(batches['image'])
        if not any([self.with_zero_centralisation, self.with_normalisation]):
            batches['image'] = self.image_rescale(batches['image'])

        return batches