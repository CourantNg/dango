# -*- coding: utf-8 -*-

'''Parse data direcotry trees and define image manipulations.'''

from dango.dataio.image.directory_parser import TREES_PARSER, ONE_HOTTOR
from dango.utilities.utility_common import task_relates_to, touch_folder
from dango.engine.default_supports import SUPPORTED_TASKS
from skimage.transform import rotate
from skimage import exposure
from random import randint
from PIL import Image
import tensorflow as tf
import numpy as np
import os


AUG_METHODS = ['flip_up_down', 'flip_left_right']


class BMProcessor(object):
    '''Block-Matching processor for GRAY images: for each pixel, given
    a reference patch around it, similar patches will be extracted in 
    search area to be representation of this pixel.'''

    def __init__(self, k=5, patchsize=3, areasize=15, threshold=40):
        '''
        param: k: int 
            number of wanted similar patches.
            take boundary into consideration, k should be smaller than
            (areasize/2)^2.
        param: patchsize: odd int
            size of each patch.
        param: areasize: odd int
            size of search area.
        param: threshold: int
            threshold.
        '''
        if patchsize % 2 != 1:
            raise ValueError("patch size must be an odd integer.")
        if areasize % 2 != 1:
            raise ValueError("search area size must be an odd integer.")
        if (areasize // 2) ** 2 < k:
            raise ValueError("no enough patches to choose.")

        self.k = k
        self.patchsize = patchsize
        self.areasize = areasize
        self.threshold = threshold

    def _hard_thresholding(self, image):
        '''Apply hard thresholding to all small patches centered at 
        every image pixel:
            patch = patch * (abs(patch) > threshold)

        param: image: np.ndarray
            image array.
        return: hard3D: np.ndarray
            holding hard thresholding vector for each pixel.
        return: img3D: np.ndarray
            holding vectorized patch for each pixel.
        '''
        size = self.patchsize
        shape = image.shape + (size ** 2,)
        hard3D = np.zeros(shape)
        img3D = np.zeros(shape)

        imagepad = np.pad(image, size // 2, 'constant')
        for i in range(shape[0]):
            for j in range(shape[1]):
                img3D[i,j,:] = imagepad[i:i+size, j:j+size].reshape(-1)
                hard3D[i,j,:] = \
                    img3D[i,j,:] * (abs(img3D[i,j,:]) > self.threshold)
    
        return hard3D, img3D

    def _get_similarities(self, ref, hardarea, imgarea):
        '''Obtain similar patches in the search area.

        param: ref: np.ndarray
            reference patch.
        param: hardarea: np.ndarray
            search area holding hard thresholding vectors.
        param: imgarea: np.ndarray
            search area holding vectorized patch.
        return: similarity: np.ndarray 
            vectorized form of these top k patches.
        '''
        hardarea = np.sum(np.square(hardarea - ref), axis=-1)
        index = np.argsort(hardarea.reshape(-1))[:self.k]
        imgarea = imgarea.reshape(-1, imgarea.shape[-1])
        tops = [imgarea[_] for _ in index]
        similarity = np.r_[tops].reshape(-1)
        return similarity

    def BMStacking(self, image):
        '''Stacking similar patches to form BM3D.

        param: image: str or np.ndarray
            input image path or input image array.
        return: BM3D: np.ndarray 
        '''
        if not isinstance(image, (str, np.ndarray)):
            raise TypeError("'image' should be str or numpy.ndarray.")
        if isinstance(image, str):
            image = np.asarray(Image.open(image).convert('L'))
        image = np.squeeze(image)
        if len(image.shape) != 2:
            raise IOError("only GRAY image can be processed now.")

        hard3D, img3D = self._hard_thresholding(image)
        shape = hard3D.shape[:2] + (self.k * self.patchsize ** 2,)
        BM3D = np.zeros(shape)

        size = self.areasize // 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                lower1 = max(0, i-size)
                lower2 = max(0, j-size)
                upper1 = min(shape[0], i+size+1)
                upper2 = min(shape[1], j+size+1)
                imgarea = img3D[lower1:upper1, lower2:upper2, :]
                hardarea = hard3D[lower1:upper1, lower2:upper2, :]
                BM3D[i,j,:] = self._get_similarities(
                    hard3D[i,j,:], hardarea, imgarea)

        return BM3D

    def locating(self):
        '''Positions for locating the wanted structure of every pixel.
        
        return: position: np.ndarray
        '''
        position = np.zeros((self.patchsize, self.patchsize, self.k))
        size = self.patchsize // 2 - 1
        if size == 0:
            position[:,:,:] = 1
        else:
            position[:, size:-size, size:-size] = 1
        position = position.reshape(-1, 1)
        
        return position


class ImageProducer(object):
    '''Parse data directory trees according to `datadir` and `task`,
    then images will be manipulated to be fed into networks based on
    `input_size` and `input_channel`.

    `datadir` has two different choices: it can be directory of 1) image 
    dataset in `datalibs` folder or 2) tfrecord files in `datarecords`
    folder. The structure of dataset and corresponding records is supposed
    to be:
    *************************************************
        -- datalibs    -- application-dataset
        -- datarecords -- application -- pickles
                                      -- records
    *************************************************

    `inference` tells which stage the network is in: inference or
    training. If in training stage, then `using_records` is an important
    indicator, which suggests whether or not convert images to tfrecords.
    `num_crxval` is the number of folds in cross-validation experiments
    if `using_records` is True; otherwise, `val_fraction` should be used 
    to indicate the fraction of dataset will be used for validation. 
    When dataset is too imbalanced, `num_instance` will be considered to
    balance dataset forcibly. `seglabels` provides labels in segmentation
    related task.

    In order to have images with desired shape, resizing, cropping or
    padding may be needed:
    1) resizing must be applied. images will be resized along a reference 
    side to preserve the original ratio between width and height. `side` 
    indicates the reference side, and `sizes` provides the desired length 
    of `side` or the desired size by (width, height).
    2) `with_crop` indicates whether cropping will be applied or not. 
    `crop_num` suggests the number of sub-images will be cropped along 
    each image side.
    3) padding will be applied if `with_pad` is True, and `min_pad`
    provides the minimal pixels to be used to do padding per time. 

    attribute: tfrecordir: str
        directory path for storing tfrecords if param `datadir` points 
        to tfrecord files; otherwise, this attribute will not exist.
    attribute: datadir: str
        directory path for storing image data if param `datadir` points 
        to image data library; otherwise, this attribute will not exist.
    other attributes can be referred to doc of __init__.
    '''

    # processors for image, like BMProcessor
    # may be used to make image representations
    bmprocessor = BMProcessor()

    def __init__(self, datadir, 
                       task, 
                       input_size=224,
                       input_channel=1,
                       inference=False,
                       num_instance=None, 
                       seglabels=None,
                       using_records=False, 
                       num_crxval=5, 
                       val_fraction=0.1,
                       sizes=[224, 256], 
                       side=None,
                       with_crop=False, 
                       crop_num=6,
                       with_pad=True, 
                       min_pad=40):
        '''
        param: datadir: str
            image data directory path or tfrecord files path.
        param: task: str
            one interested task in SUPPORTED_TASKS.
        param: input_size: int, optional
            image size will be fed into network.
        param: input_channel: int, optional
            image channel will be fed into network.
        param: inference: bool, optional
            indicating whether or not it's in inference stage.
        param: num_instance: int or None, optional
            indicating the number of instances to be used for each label.
            If `None`, then all instances will be used.
        param: seglabels: list or tuple or None, optional
            containing all pixel values to be labels for segmentation 
            related task. For example, a gray mask image with only 0 
            and 255 pixel values has two labels: 0 and 255.
        param: using_records: bool, optional
            indicating whether using tfrecords to provide data.
        param: num_crxval: int, optional
            number of folds in cross validation if `using_records` is True.
        param: val_fraction: float, optional
            fraction for validation in cross validation if `using_records` 
            is False. 
        param: sizes: list, optional
            if it's a list, then each element represents an image size, and
            can be an integer, a list or a tuple.
        param: side: 'longer', 'shorter' or None, optional
            indicating which side as the reference side.
        param: with_crop: bool, optional
            indicating whether images will be cropped.
        param: crop_num: int, optional
            number of cropped sub-images along each side.
        param: with_pad: bool, optional
            indicating whether images will be padded.
        param: min_pad: int, optional
            the minimal number of pixels for padding per time.
        '''
        self.task = task
        self.input_size = input_size
        self.input_channel = input_channel
        self.seglabels = seglabels

        self.using_records = using_records
        self.num_instance = num_instance
        if self.using_records: self.num_crxval = num_crxval
        else: self.val_fraction = val_fraction

        self.sizes = sizes
        self.side = side
        self.with_crop = with_crop
        self.crop_num = crop_num
        self.with_pad = with_pad
        self.min_pad = min_pad

        self.inference = inference
        self._check(datadir=datadir)
        self._one_hotting()

    def _look_up_path(self, wanted='pickles'):
        '''Look up sub-folder's path(will be created if not exist) 
        in folder 'datarecords.app' based on the choice of `wanted`: 
        1) `pickles`: for storing pickle files; 
        2) `records`: for storing tfrecord files.

        param: wanted: str
            one of 'pickles' and 'records'.
        return: absolute path of 'pickles' or 'records'.
        '''
        if hasattr(self, 'datadir'): # image data library
            application_name = self.datadir.rsplit(os.sep, 1)[-1]
            path = os.path.dirname(os.path.abspath(self.datadir)) # datalibs
            path = os.path.join(os.path.dirname(path), 
                'datarecords', application_name, wanted) # datarecords.app.records
            return touch_folder(path)
        if hasattr(self, 'tfrecordir'): # tfrecord files directory
            return os.path.join(os.path.dirname(
                os.path.abspath(self.tfrecordir)), wanted)

    def _check(self, datadir):
        '''Check reasonability of values of attributes.
        
        param: datadir: str
            image data directory or tfrecord files path.
        '''
        # check `datadir` or `tfrecordir` attribute
        # if `datadir` is tfrecords path, then the existence of tfrecod 
        # files and pickle files will be checked.
        if datadir.rsplit(os.sep, 1)[-1] == 'records': # tfrecords path
            if len([_file for _file in os.listdir(datadir) 
                if '.tfrecords' in _file]) <= 0:
                raise IOError("no threcord files exist.")
            self.tfrecordir = datadir

            _pickles = self._look_up_path()
            if (not os.path.exists(_pickles) or 
                len(os.listdir(_pickles)) <= 0):
                raise IOError("pickles must exist when using tfrecords.")
            
            self.using_records = True 
            assert hasattr(self, 'num_crxval')
        else: # image data directory
            if not os.path.isdir(datadir) or len(os.listdir(datadir)) <= 0:
                raise IOError("not found data in '{}'.".format(datadir))
            self.datadir = datadir

        # check `task``
        if self.task not in SUPPORTED_TASKS:
            raise ValueError("unrecognized task {}.".format(self.task))

        # check `seglabels`
        if task_relates_to(self.task, 'segmentation'):
            if (not isinstance(self.seglabels, (tuple, list)) 
                or len(self.seglabels) <= 0):
                raise ValueError("no labels provided for segmentation.")

        # check `with_crop` and `with_pad`: 
        # these two things are not consistent
        if all([self.with_crop, self.with_pad]):
            raise ValueError("'with_crop' and 'with_pad' should not " 
                "take True at the same time.")

        # check `input_size` and `input_channel`
        assert isinstance(self.input_channel, int)
        assert isinstance(self.input_size, (int, tuple, list))
        if isinstance(self.input_size, int):
            self.input_size = [self.input_size] * 2
        else:
            self.input_size = list(self.input_size)

    def _one_hotting(self):
        '''One hotting is needed only when pickle files doesn't 
        exist at the first training period.'''
        path = self._look_up_path()
        pickles = [_file for _file in os.listdir(path)
            if '.pk' in _file]

        if (not self.inference and hasattr(self, 'datadir') 
            and len(pickles) == 0):
            if task_relates_to(self.task, 'classification'):
                ONE_HOTTOR['classification'](self.datadir, path)
            if task_relates_to(self.task, 'segmentation'):
                ONE_HOTTOR['segmentation'](self.seglabels, path)
        if self.inference or hasattr(self, 'tfrecordir'):
            assert len(pickles) > 0, "not found needed pickle files."

    def _parsing_directory_trees(self):
        '''Group images(and corresponding labels or masks, where 
        `label` corresponds to image classification label and `mask`
        corresponds to image segmentation mask.)

        The structure of the result of parsing directory trees by 
        TREES_PARSER, which can be seen as the input of this function, 
        is supposed to be like:
            {
                'lable-1': 
                    [[image-1, mask-1], [image-2, mask-2]],
                'lable-2': 
                    [[image-1, mask-1], [image-2, mask-2]],
            }
        (in segmentation related task, each `instance` is `[image, mask]`;
        in classification task, each `instance` is just a str.)

        return: crxvals: list or dict according to `using_records`
            if `using_records` is True, then `crxvals` is
                [dict-1, dict-2, ...]
                dict-i = {
                    'label-1': [...], 'label-2': [...]
                }
            if `using_records` is False, then `crxvals` is
                {
                    'train': 
                        {'label-1': [...], 'label-2': [...]}, 
                    'validate': 
                        {'label-1': [...], 'label-2': [...]}
                }
        '''
        assert hasattr(self, 'datadir')
        images = TREES_PARSER[self.task](self.datadir)

        if self.using_records: 
            crxvals = [dict() for _ in range(self.num_crxval)]
        else:
            crxvals = {'train': dict(), 'validate': dict()}

        for label in images.keys():
            if self.num_instance:
                images[label] = images[label][:self.num_instance]
            else:
                self.num_instance = len(images[label])

            if self.using_records:
                num_instance = self.num_instance // self.num_crxval
                for index, _dict in enumerate(crxvals):
                    _dict[label] = list()
                    lower = num_instance * index
                    upper = num_instance * (index + 1)
                    _dict[label] = images[label][lower:upper]
            else:
                crxvals['train'][label] = list()
                crxvals['validate'][label] = list()
                validate_num = int(self.num_instance * self.val_fraction)
                crxvals['validate'][label] = images[label][:validate_num]
                crxvals['train'][label] = images[label][validate_num:]
                if validate_num == 0: crxvals.pop('validate')
        
        return crxvals

    @staticmethod
    def image_open(image, mode=None):
        '''Load image from str or bytes to PIL.Image.

        param: image: str or bytes
            image path.
        param: mode: None or str or int, optional
            one of None, 'L', 'RGB', 1 and 3.
        return: image: PIL.Image
            image object.
        '''
        if not isinstance(image, (str, bytes)):
            raise TypeError("'image' should be 'str' or 'bytes'.")
        if isinstance(image, bytes):
            image = image.decode('utf-8')

        image = Image.open(image)
        if mode is not None:
            assert mode in ['L', 'RGB', 1, 3]
            if isinstance(mode, int):
                mode = 'L' if mode == 1 else 'RGB'
            image = image.convert(mode)

        return image

    @staticmethod
    def image_array_show(image):
        '''Display image array with shape (#, #) or (#, #, 3).
        
        param: image: np.ndarray
        '''
        assert isinstance(image, np.ndarray)
        image = np.squeeze(image.astype(np.uint8))
        image.show()

    @staticmethod
    def image_resize(image, size, mask=None, side=None):
        '''Resize image based on `size` and `side`. If `mask` image is
        also provided, then it will be resized, too.

        1) `size` is an integer:
            If 'side' is None, then both shorter and longer side will be 
            resized to match `size`; otherwise, `side` is one of 'shorter'
            and 'longer', and in this case, that side will be resized to 
            match `size`, and the rest side will be:
                new-rest-side = (size * rest-side) / side
        2) `size` is a list or a tuple: 
            resize directly and `size` should be `(width, height)`.

        param: image: PIL.Image
            image object.
        param: size: int, list or tuple
            desired size.
        param: mask: PIL.Image or None
            mask image object.
        param: side: str or None, optional
            reference side, 'shorter' or 'longer' or None.
        return: image: PIL.Image
        return: mask: PIL.Image or None
        '''
        if not isinstance(size, (int, list, tuple)):
            raise TypeError("'size' should be one of int, list or tuple.") 
        if mask is not None: assert image.size == mask.size

        if isinstance(size, int):
            if side is None:
                image = image.resize(size=(size, size))
                if mask is not None: mask = mask.resize(size=(size, size))
                return image, mask

            assert side in ['shorter', 'longer']
            (width, height) = image.size

            cond1 = height > width and side == 'shorter'
            cond2 = height <= width and side == 'longer'
            if cond1 or cond2: 
                height, width = (height * size) // width, size
            else: 
                height, width = size, (width * size) // height
            
            image = image.resize(size=(width, height))
            if mask is not None: mask = mask.resize(size=(width, height))
        else:
            image = image.resize(size=tuple(size))
            if mask is not None: mask = mask.resize(size=tuple(size))
                
        return image, mask

    @staticmethod
    def image_pad(image, mask=None, min_pad=40, mode='constant'):
        '''Pad image based on `mode` and `min_pad`. If `mask` image
        is also provided, then it will be padded, too.

        param: image: PIL.Image or numpy.ndarray
            image.
        param: mask: PIL.Image object or numpy.ndarray or None
            mask image.
        param: min_pad: int, optional
            the minimal number pixels for padding per time.
        param: mode: str, optional
            pad-mode(refer to doc of 'numpy.pad' for more info).
            if `mode` is `constant`, then it will be padded by 0,
            which is recommended in most cases.
        return: image: numpy.array
            image array.
        return: mask: numpy.array or None
            mask image array.
        '''
        image = np.squeeze(np.asarray(image))
        if mask is not None: 
            mask = np.squeeze(np.asarray(mask))
            assert len(mask.shape) == 2, "mask should always be GRAY format." 
            assert image.shape[:2] == mask.shape, "image and mask mismatch."
        height, width = image.shape[0], image.shape[1]
        channels = 1 if len(image.shape) == 2 else 3

        if height == width: return [(image, mask)]

        while(height != width):
            differ = min(abs(height - width), min_pad)
            if differ % 2 == 0: pad_with = (differ // 2, differ // 2)
            else: pad_with = (differ // 2, differ // 2 + 1)

            if height > width:
                mask_pad_with = ((0, 0), pad_with)
                if channels == 3: pad_with = ((0, 0), pad_with, (0, 0))
                if channels == 1: pad_with = ((0, 0), pad_with)
            else:
                mask_pad_with = (pad_with, (0, 0))
                if channels == 3: pad_with = (pad_with, (0, 0), (0, 0))
                if channels == 1: pad_with = (pad_with, (0, 0))

            image = np.pad(image, pad_with, mode)
            height, width = image.shape[0], image.shape[1]
            if mask is not None: 
                mask = np.pad(mask, mask_pad_with, mode)
                assert (height, width) == mask.shape

        assert image.shape[0] == image.shape[1]
        if mask is not None: assert mask.shape[0] == mask.shape[1]

        return [(image, mask)]

    @staticmethod
    def image_crop(image, size, mask=None, num=6, 
                   once=False, offset=None):
        '''Crop images based on `size`, `num`, `once` and `offset`.
        If `mask` is provided, then it will be cropped in the same way.

        param: image: PIL.Image
            image.
        param: size: int, list or tuple
            size for cropping. When 'size' is a list or tuple, 
            it should be (width, height).
        param: mask: PIL.Image or None, optional
            mask image.
        param: num: int, optional
            number of cropped sub-images wanted per side.
        param: once: bool, optional
            indicating whether just crop once.
        param: offset: None or tuple, optional
            if it's a tuple, then it offers the start point for cropping
            when 'once' is True. Also, it should be (width, height).
        return: images: list
            element is (cropped_image, cropped_mask) or
            (cropped_image, None), where cropped_image and/or cropped_mask
            is(are) PIL.Image object.
        return: offset: tuple or None
        '''
        if not isinstance(size, (int, list, tuple)):
            raise TypeError("'size' should be int, list or tuple.")
        if isinstance(size, (tuple, list)):
            _width, _height = tuple(size)
        else:
            _width, _height = size, size

        images = list() # returned
   
        width, height = image.size
        if width == _width: offset_width = 0
        else: offset_width = width - _width - 1
        if height == _height: offset_height = 0
        else: offset_height = height - _height - 1

        if once:
            if offset:
                assert isinstance(offset, tuple)
                offset_width, offset_height = offset
            else:
                offset_width = randint(0, offset_width) 
                offset_height = randint(0, offset_height)
            box = (offset_width, offset_height, 
                offset_width + _width, offset_height + _height)
            _image = image.crop(box)
            if mask is not None: 
                images.append((_image, mask.crop(box)))
            else: images.append((_image, None))

            return images, (offset_width, offset_height)

        width_stride = offset_width // num
        height_stride = offset_height // num
        if width_stride == 0: (offset_width, width_stride) = (1, 1)
        if height_stride == 0: (offset_height, height_stride) = (1, 1)

        for i in range(0, offset_width, width_stride):
            for j in range(0, offset_height, height_stride):
                box = (i, j, i + _width, j + _height)
                _image = image.crop(box)
                if mask is not None: 
                    images.append((_image, mask.crop(box)))
                else: images.append((_image, None))

        return images, None

    @staticmethod
    def image_transform(image, mask=None):
        '''Transform image according to AUG_METHODS.

        param: image: numpy.ndarray
            input image array. 
        param: mask: numpy.ndarray or None
            mask image.
        return: image: numpy.ndarray
            image after transforming.
        return: mask: numpy.ndarray or None
            mask image after transforming.
        '''
        image = np.squeeze(image)
        if mask is not None: mask = np.squeeze(mask)

        index = randint(0, len(AUG_METHODS))
        if index != len(AUG_METHODS):
            if AUG_METHODS[index] == 'flip_up_down':
                image = np.flipud(image)
                if mask is not None: mask = np.flipud(mask)

            if AUG_METHODS[index] == 'flip_left_right':
                image = np.fliplr(image)
                if mask is not None: mask = np.fliplr(mask)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if mask is not None and len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        return image, mask

    @staticmethod
    def image_transform_ingraph(image, mask):
        '''Image transformation in graph version.

        example:
        if mask: image, mask = image_transform_ingraph(image, mask)
        else: image, _ = image_transform_ingraph(image, image)

        param: image: tensorflow.Tensor
            image tensor.
        param: label: tensorflow.Tensor
            mask image tensor.
        return: image: tensorflow.Tensor
            image tensor after transforming.
        return: mask: tensorflow.Tensor or None
            mask image after transformaing.
        '''
        pred_fn_pairs = dict()
        rand_num = tf.random_uniform(shape=(), minval=0,
            maxval=len(AUG_METHODS) + 1, dtype=tf.int32)
        
        # 0 <--> 'flip_up_down'
        pred_fn_pairs[tf.equal(rand_num, 0)] = \
            lambda: (tf.image.flip_up_down(image),
                tf.image.flip_up_down(mask))
        # 1 <--> 'flip_left_right'
        pred_fn_pairs[tf.equal(rand_num, 1)] = \
            lambda: (tf.image.flip_left_right(image),
                tf.image.flip_left_right(mask))
        # 2 <--> 'original'
        pred_fn_pairs[tf.equal(rand_num, 2)] = lambda: (image, mask)

        image, mask = tf.case(pred_fn_pairs, exclusive=True)
        return image, mask

    @staticmethod
    def image_equalize_hist(image):
        '''Return image after histogram equalization.

        param: image: str or numpy.ndarray
            image path or image array.
        return: image: numpy.ndarray
            image array after histogram equalization.
        '''
        if not isinstance(image, (str, bytes, np.ndarray)):
            raise TypeError("'image' should be str, bytes or np.ndarray.")
        if isinstance(image, (str, bytes)):
            image = ImageProducer.image_open(image, 'L')
        image = np.squeeze(np.asarray(image, dtype=np.int32))
        assert len(image.shape) == 2

        image = (exposure.equalize_hist(image) * 255).astype(np.int32)
        image = np.expand_dims(image, axis=-1) # 3-dim

        return image

    @staticmethod
    def image_rotate(image, angle, dtype=np.float):
        '''Image rotation.

        param: image: numpy.ndarray, str or bytes
            image path or image array.
        param: angle: float
            angle for ratation.
        return: image: numpy.ndarray.
            image array.
        '''
        if not isinstance(image, (str, bytes, np.ndarray)):
            raise TypeError("'image' should be str, bytes or np.ndarray.")
        if isinstance(image, (str, bytes)):
            image = ImageProducer.image_open(image)
        else:
            image = Image.fromarray(np.squeeze(image))

        image = image.rotate(angle)
        image = np.asarray(image, dtype=dtype)
        if len(image.shape) == 2: 
            image = np.expand_dims(image, axis=-1) # 3-dim
        
        return image

    @staticmethod
    def image_zero_centralization(images, axis=0):
        '''Zero cnetralization. 

        param: images: numpy.ndarray
            images.
        param: axis: int or tuple of int.
            0 or (0, 1, 2).
        return: images: numpy.ndarray
            images.
        '''
        images = np.asarray(images, dtype=np.float)
        images -= np.mean(images, axis=axis)
        return images

    @staticmethod
    def image_normalisation(images):
        '''Normalisation(including PCA and whitening) is 
        not common in image processing. 

        If an element of `std` is zero, which suggests that 
        this element has no deviation, it is an useless feature.

        param: images: numpy.ndarray
            images.
        return: images: numpy.ndarray
            images.
        '''
        images = np.asarray(images, dtype=np.float)
        images /= (np.std(images, axis=0) + 1e-5) 
        return images

    @staticmethod
    def image_rescale(images):
        '''Rescale images to [-0.5, 0.5].'''
        images = (images / 255) # - 0.5 
        return images

    @staticmethod
    def image_bm_processor(image):
        '''Block-Matching processor for gray image. Refer to the doc
        of `BMProcessor` for more information.
        
        param: image: str or np.ndarray
            input image path or input image array.
        return: BM3D: np.ndarray 
        '''
        return ImageProducer.bmprocessor.BMStacking(image)
        