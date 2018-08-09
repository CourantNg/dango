# -*- coding: utf-8 -*-

from PIL.Image import fromarray
import tensorflow as tf

class TFRecordVisulizer(object):
    '''Two functionalities:
        1) check tfrecords.
        2) basic usage of ImageProvider.
    '''

    def __init__(self, image_provider, iteration=2, batch=5):
        '''
        param: image_provider:
            `dango.dataio.image.image_provider.ImageProvider` object.
        param: iteration: int
            iteration number.
        param: batch: int
            batch size.
        '''
        self.image_provider = image_provider
        self.iteration = iteration
        self.batch = batch

    @property
    def records_num(self):
        '''Count the total number of records.'''
        tfrecords = list()
        for state in self.image_provider.records.keys():
            tfrecords += self.image_provider.records[state]
        
        number = 0
        for tfrecord in tfrecords:
            number += sum([1 for _ in 
                tf.python_io.tf_record_iterator(tfrecord)])
        return number

    def visualize(self):
        tensors = self.image_provider.next_data(batch=self.batch)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            for _ in range(self.iteration):
                batches = self.image_provider.convert_to_array(
                    tensors['train'], sess)
                
                if batches.get('classification', None) is not None:
                    print(batches['classification'])
                if batches.get('segmentation', None) is not None:
                    print(batches['segmentation'].shape)
                
                for i in range(self.batch):
                    image = batches['image'][i]
                    self.image_provider.image_array_show(image)
                    try:
                        image = batches['segmentation'][i]
                        self.image_provider.image_array_show(image)
                    except:
                        pass

            coord.request_stop()
            coord.join(threads)