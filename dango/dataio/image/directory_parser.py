# -*- coding: utf-8 -*-

'''Parse image data directory trees.'''

from dango.utilities.utility_common import touch_folder
from random import shuffle
from pickle import dump
import numpy as np
import os

IMAGE_FORMATS = ['.jpg', '.png', '.bmp']
IMAGE_FORMATS += [_.upper() for _ in IMAGE_FORMATS]
LOSSLESS_IMAGE_FORMATS = ['.png', '.bmp']
LOSSLESS_IMAGE_FORMATS += [_.upper() for _ in LOSSLESS_IMAGE_FORMATS]


def isimage(name, formats=IMAGE_FORMATS):
    '''To judge whether or not a file is an image.'''
    suffix = os.path.splitext(name)[-1]
    if suffix in formats: return True
    return False


def collect_segpair(path):
    '''Collect image and corresponding mask image in the `path`-dir 
    to an empty list in segmentation related task.

    param: path: str
        path for storing original and mask image.
    return: instance: list
        containing image path and corresponding mask image path.
    '''
    _instance = list()
    for name in os.listdir(path):
        if (os.path.splitext(name)[0] == 'original'
            and isimage(name)):
            _instance.append(name)
        if (os.path.splitext(name)[0] == 'mask' and 
            isimage(name, LOSSLESS_IMAGE_FORMATS)):
            _instance.append(name)
    
    # ensure to get [ori, mask]
    _instance = sorted(_instance, reverse=True) 
    if len(_instance) != 2: 
        raise IOError("no enough or too many data "
            "in path '{}'.".format(path))

    _instance = [os.path.join(path, _) for _ in _instance]
    return _instance
    

class ParseDirectoryTrees(object):
    '''Parsing directory trees for different tasks.

    The directory structure of the image dataset is supposed to be:
    --------------------------------------------------------------------
    data-dir-----(label-1)----(instance-1)--original-image(, mask-image)
              |		   |------(instance-2)--original-image(, mask-image)
              |
              |--(label-2)----(instance-1)--original-image(, mask-image)
                       |------(instance-2)--original-image(, mask-image)
    --------------------------------------------------------------------

    1) If it's a classification task, `instance` folder, denoted by 
    `(instance-1)`-like, will be omitted.
    
    2) If it's a segmentation task, `label` folder, denoted by 
    `(label-1)`-like, will be omitted. 
    '''

    @staticmethod
    def parsing_classification_tree(datadir):
        '''Parse directory trees for classification tasks. Note that 
        `instance` folders are omitted in the data directory trees.

        param: datadir: str
            path of the directory trees.
        return: images: dict
            {
                'label-1': 
                    [image-1, image-2], 
                'label-2': 
                    [image-1, image-2]
            }.
        '''
        images = dict()
        for label in os.listdir(datadir):
            images[label] = list()
            label_path = os.path.join(datadir, label)
            for image_name in os.listdir(label_path):
                if isimage(image_name):
                    images[label].append(os.path.join(
                        label_path, image_name))
            shuffle(images[label])

        return images

    @staticmethod
    def parsing_segmentation_tree(datadir):
        '''Parse directory trees for segmentation tasks. Note that 
        `label` folders are omitted in the data directory trees.

        param: datadir: str
            path of the directory trees.
        return: images: dict
            a dictionary with a key 'label' just for consistence
            with results of other parsing functions.
                {
                    'label': 
                        [[image-1, mask-1], 
                        [image-2, mask-2]]
                }.
        '''
        images = {'label': list()}
        for instance in os.listdir(datadir):
            instance_path = os.path.join(datadir, instance)
            _instance = collect_segpair(instance_path)
            images['label'].append(_instance)
        shuffle(images['label'])

        return images

    @staticmethod
    def parsing_classification_segmentation_tree(datadir):
        '''Parse directory trees for classification-segmentation task.

        param: datadir: str
            path of the directory trees.
        return: images: dict
            {
                'label-1': 
                    [(image-1, mask-1), (image-2, mask-2)],
                'label-2': 
                    [(image-1, mask-1), (image-2, mask-2)]
            }.
        '''
        images = dict()
        for label in os.listdir(datadir):
            images[label] = list()
            label_path = os.path.join(datadir, label)
            for instance in os.listdir(label_path):
                instance_path = os.path.join(label_path, instance)
                _instance = collect_segpair(instance_path)
                images[label].append(_instance)
            shuffle(images[label])

        return images

    @staticmethod
    def classification_one_hotting(datadir, savedir):
        '''One hotting codes in classification related task: each 
        label relates to a 'np.ndarray' with `(1, #labels)` shape.

        A dict with two sub-dicts will be saved:
        `classification-one-hotting` is defined by
            key   <--> label
            value <--> corresponding one-hotting code
        and `classification-label` is defined by
            key   <--> the argmax of an one-hotting code
            value <--> corresponding label

        param: datadir: str
            path of the data directory trees.
        param: savedir: str
            path to save pickle file: classification.pk.
        '''
        touch_folder(savedir)
        _pickle = {
            'classification-one-hotting': dict(), 
            'classification-label': dict()
        }

        labels = os.listdir(datadir)
        label_num = len(labels)

        for index, label in enumerate(labels):
            _pickle['classification-one-hotting'][label] = \
                np.zeros([1, label_num])
            _pickle['classification-one-hotting'][label][0][index] = 1
            _pickle['classification-label'][index] = label

        save_path = os.path.join(savedir, 'classification.pk')
        with open(save_path, 'wb') as fp: dump(_pickle, fp)

    @staticmethod
    def segmentation_one_hotting(seglabels, savedir):
        '''One-hotting codes in segmentation related tasks.

        A dict with two sub-dicts will be saved:
        'segmentation-one-hotting':
            key   <--> segmentation label
            value <--> corresponding one-hotting code
        'segmentation-label':
            key   <--> the argmax of an one-hotting code
            value <--> corresponding segmentation label

        param: seglabels: list
            containing all pixel values to be one-hotting.
        param: savedir: str
            path to save pickle file: segmentation.pk.
        '''
        touch_folder(savedir)
        _pickle = {
            'segmentation-one-hotting': dict(), 
            'segmentation-label': dict()
        }

        num_labels = len(seglabels)

        for index, label in enumerate(seglabels):
            _pickle['segmentation-one-hotting'][label] = \
                np.zeros([1, num_labels], dtype=np.int32) # 
            _pickle['segmentation-one-hotting'][label][0][index] = 1
            _pickle['segmentation-label'][index] = label

        save_path = os.path.join(savedir, 'segmentation.pk')
        with open(save_path, 'wb') as fp: dump(_pickle, fp)


TREES_PARSER = {
    'classification': 
        ParseDirectoryTrees.parsing_classification_tree,
    'segmentation': 
        ParseDirectoryTrees.parsing_segmentation_tree,
    'classification-segmentation':
        ParseDirectoryTrees.parsing_classification_segmentation_tree
}


ONE_HOTTOR = {
    'classification': 
        ParseDirectoryTrees.classification_one_hotting,
    'segmentation': 
        ParseDirectoryTrees.segmentation_one_hotting
}