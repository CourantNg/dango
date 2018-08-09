# -*- coding: utf-8 -*-

'''Convert images and corresponding labels or/and masks 
to records.'''

from dango.dataio.image.image_producer import ImageProducer
from dango.utilities.utility_common import task_relates_to
from dango.utilities.utility_common import clean_folder
from dango.utilities.utility_common import time_duration
from threading import Thread
from PIL import Image
import tensorflow as tf
import numpy as np
import os


class RecordsGenerator(ImageProducer):
    '''Generate records based on `crxvals` defined in 
    `ImageProducer._parsing_directory_trees`.'''

    def __init__(self, datadir, 
                       task,
                       input_size=224, 
                       input_channel=1,
                       num_instance=None,
                       seglabels=None,
                       num_crxval=5,
                       sizes=[224, 256], 
                       side=None,
                       with_crop=False, 
                       crop_num=6,
                       with_pad=True, 
                       min_pad=40):
        '''Refer to doc of `ImageProducer` defined in 
        `dango.dataio.image.image_producer`.'''
        super(RecordsGenerator, self).__init__(
            datadir=datadir, 
            task=task,
            input_size=input_size, 
            input_channel=input_channel,
            num_instance=num_instance, 
            seglabels=seglabels,
            using_records=True, 
            num_crxval=num_crxval,
            sizes=sizes, 
            side=side,
            with_crop=with_crop, 
            crop_num=crop_num,
            with_pad=with_pad, 
            min_pad=min_pad
        )

    def _get_bytes(self, image_path, label=None, mask=None):
        '''Obtain image_bytes, label_bytes and mask_bytes. Images and 
        corresponding masks and/or labels will be resized firstly, then
        may be padded or cropped, and finally will be converted to bytes.

        param: image: str
            image path.
        param: label: str or None, optional
            image label.
        param: mask: str or None, optional
            mask image path.
        return: bytes_list: list
            each elemet is a dict
            {   
                'image': image-bytes, # must have
                'classification': label-bytes, # may have
                'segmentation': mask-bytes # may have
            }
        '''
        image = self.image_open(image_path, self.input_channel)
        if mask is not None: 
            mask = self.image_open(mask)
            if image.size != mask.size:
                raise IOError("shape mismatch: {}.".format(image_path))

        images = list()
        for size in self.sizes:
            _image, _mask = self.image_resize(image, size, mask, self.side)
            if self.with_crop:
                _images, _ = self.image_crop(
                    _image, self.input_size, _mask, self.crop_num)
            elif self.with_pad:
                _images = self.image_pad(_image, _mask, self.min_pad)
            else:
                _images = [(_image, _mask)]
            images.extend(_images)

        bytes_list = list()
        for image, mask in images:
            _byte = dict()
            image = np.squeeze(np.asarray(image)).astype(np.uint8)
            _byte['image'] = image.tobytes()
            if label is not None: 
                _byte['classification'] = label.encode('utf-8')
            if mask is not None: 
                mask = np.squeeze(np.asarray(mask)).astype(np.uint8)
                _byte['segmentation'] = mask.tobytes()
            bytes_list.append(_byte)

        return bytes_list

    def _convert_to_record(self, image, label=None, mask=None):
        '''Convert an image and its corresponding label and/or mask 
        to a record, i.e. serialized example.

        param: image: str
            image path.
        param: label: str or None, optional
            image classification label.
        param: mask: str or None, optional
            image segmentation mask.
        return: examples: list
            a list with serialized example.
        '''
        bytes_list = self._get_bytes(image, label, mask)

        examples = list()
        for _byte in bytes_list:
            feature = dict()
            for _key in _byte.keys():
                bytes_list = tf.train.BytesList(value=[_byte[_key]])
                feature[_key] = tf.train.Feature(bytes_list=bytes_list)
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            examples.append(example)

        return examples

    def _generate_records(self, save_path, crxval_fold):
        '''Generate tfrecord files for a crxval fold.

        For segmentation related task, `instance` is `(image, mask)`,
        and `label` is needed in classification-segmentation task, or
        can be omitted in segmentation task.

        For classification task, `instance` is just a str, `mask` can 
        be omitted.

        param: save_path: str
            path for saving tfrecord files.
        param: crxval_fold: tuple
            (crxval_fold_dict, crxval_fold_index).
        yield: tfrecord files for this crxval fold.
        '''
        save_path = os.path.join(save_path, 
            "%.5d-of-%.5d.tfrecords" % (crxval_fold[1], self.num_crxval))
        with tf.python_io.TFRecordWriter(save_path) as writer:
            for label in crxval_fold[0].keys():
                for instance in crxval_fold[0][label]:
                    if task_relates_to(self.task, 'segmentation'):
                        assert isinstance(instance, (tuple, list))
                        image, mask = instance[0], instance[1]
                        examples = self._convert_to_record(image, label, mask)
                    if self.task == 'classification':
                        assert isinstance(instance, str)
                        examples = self._convert_to_record(instance, label)
                    for example in examples:
                        writer.write(example.SerializeToString())

    @time_duration
    def generate_records(self):
        '''Generate records for all images and their corresponding 
        labels and/or masks.'''
        assert hasattr(self, 'datadir')
        save_path = clean_folder(self._look_up_path('records'))
        crxvals = self._parsing_directory_trees()

        threads = list()
        for index, _dict in enumerate(crxvals):
            thread_item = Thread(
                target = self._generate_records,
                args = (save_path, (_dict, index)))
            thread_item.start()
            threads.append(thread_item)

        for thread_item in threads: thread_item.join()

    def __call__(self):
        self.generate_records()