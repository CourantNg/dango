# -*- coding: utf-8 -*-

'''This module defines a general procedure for running networks for
different tasks.

Basic usage:
    driver = Driver(arguments)
    driver.run()
where `arguments` are generated from `dango.utilities.parameters_parser.parse`.
'''

from dango.engine.default_supports import SUPPORTED_BASE_TASKS
from dango.utilities.utility_common import task_relates_to
from dango.engine.loader import Loader
import tensorflow as tf
import os


class Driver(Loader):
    '''Driver for running networks.'''

    def __init__(self, arguments):
        super(Driver, self).__init__(arguments=arguments)

    def _stdout_msgs(self, iteration, train_eval, validate_eval):
        '''Print training and validating performance.

        param: iteration: int
            iteration.
        param: train_eval: dict
            {'classification', 'segmentation'}.
        param: validate_eval: dict
            {'classification', 'segmentation'}.
        '''
        messages = ['CrxVal:%d' % (self.crxval_index + 1)]
        messages.append('iter:%d' % iteration)

        for task in SUPPORTED_BASE_TASKS:
            if train_eval.get(task, None) is not None:
                messages.append("{}: train:{}".format(
                    task, str(train_eval[task])))
                if (validate_eval is not None and
                    validate_eval.get(task, None) is not None):
                    messages.append("val:{}".format(
                        str(validate_eval[task])))

        tf.logging.info(', '.join(messages))

    def _update_network(self, batches, sess):
        '''Optimize network.'''
        feed_dict = {
            self.network.input_images: batches['image'],
            self.network.eval_training: True,
            self.network.training: True
        }
        for task in SUPPORTED_BASE_TASKS:
            if task_relates_to(self.network.task, task):
                feed_dict[self.network.gts[task]] = batches[task]
        sess.run(self.network.optimizations, feed_dict=feed_dict)

    def _eval_network_v1(self, batches, eval_training, sess, 
                            options=None, run_metadata=None):
        '''Network evaluation.

        param: batches: dict
            {'image', 'classification', 'segmentation'}
        param: eval_training: bool
            indicating whether to evaluate training performance.
        return: evaluation: dict
            {'classification', 'segmentation'}
        return: summary_protocol:
            summary protocol.
        '''
        feed_dict = {
            self.network.input_images: batches['image'],
            self.network.eval_training: eval_training,
            self.network.training: False
        }
        for task in SUPPORTED_BASE_TASKS:
            if self.network.evaluation.get(task, None) is not None:
                feed_dict[self.network.gts[task]] = batches[task]

        evaluation, merged = sess.run(
            [self.network.evaluation, self.network.merged],
            feed_dict = feed_dict, options=options, 
            run_metadata=run_metadata)
        return evaluation, merged

    def _eval_network_v2(self, batches, eval_training, 
                               sess, writer, iteration):
        '''Evaluate network. 

        param: batches: dict
            {'image', 'classification', 'segmentation'}
        param: eval_training: bool
            indicating whether to evaluate training performance.
        return: evaluation: dict
            {'classification', 'segmentation'}
        '''
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        evaluation, merged = self._eval_network_v1(
            batches, eval_training=eval_training, sess=sess,
            options=run_options, run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'setp%5d' % iteration)
        writer.add_summary(merged, iteration)
        self._record_timeline(run_metadata, iteration)
        
        return evaluation

    def _train_network(self, key=None):
        '''Train network.
        
        param: key: str
            keyword for fetching variables to restore in
            `Loader._fetch_variables_to_restore`.
        '''
        tensors = self.data_provider.next_data(self.batch)

        train_writer = Loader._summaries_writer(
            'train', self.logs_savedir)
        if tensors.get('validate', None) is not None:
            validate_writer = Loader._summaries_writer(
                'validate', self.logs_savedir)
        
        saver, variables_to_initialise = self._load_saver(key) # key = 'seg'

        with tf.Session(config=self.config) as sess:
            train_writer.add_graph(sess.graph)
            validate_writer.add_graph(sess.graph)

            self._init_network(sess, saver, variables_to_initialise)  
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.graph.finalize() # graph read-only
            tf.logging.info('starting training network...')

            for iteration in range(self.from_iteration, self.to_iteration):
                try:
                    batches = self.data_provider.convert_to_array(
                        tensors['train'], sess)
                    self._update_network(batches, sess)

                    if iteration % self.tensorboard_every_n == 0:
                        train_eval = self._eval_network_v2(batches,
                            True, sess, train_writer, iteration)
                    
                    if (iteration % self.validation_every_n == 0
                        or iteration == self.from_iteration):
                        if tensors.get('validate', None) is not None:
                            batches = self.data_provider.convert_to_array(
                                    tensors['validate'], sess)
                            validate_eval = self._eval_network_v2(batches, 
                                False, sess, validate_writer, iteration)
                        else:
                            validate_eval = None

                        self._stdout_msgs(iteration, train_eval, validate_eval)

                    if iteration % self.save_every_n == 0:
                        self._make_checkpoint(saver, sess, iteration)
                
                except KeyboardInterrupt:
                    self._make_checkpoint(saver, sess, iteration)

            coord.request_stop()
            coord.join(threads)

        train_writer.close()
        validate_writer.close()

    def _predict_network(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self._init_network(sess, saver)
            while True:
                data = self.data_provider.next_data()
                if data is None: 
                    break
                else: # data = (images, {'name', 'path'})
                    feed_dict = {
                        self.network.input_images: data[0],
                        self.network.training: False
                    }
                    logits = sess.run(self.network.logits, 
                        feed_dict=feed_dict)
                    self._interpret_outputs[self.task](logits, data[1])

    def run(self):
        try:
            if self.action == 'train': self._train_network()
            else: self._predict_network()
        except KeyboardInterrupt:
            tf.logging.info('stop running by user.')
            exit()