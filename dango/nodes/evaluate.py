# -*- coding: utf-8 -*-

'''Evaluation.'''

import tensorflow as tf


def accuracy_evaluator(logits, labels):
    '''Accuracy evaluation.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor 
        ground truth.
    return: accuracy: tensorflow.Tensor 
        accuracy.
    '''
    with tf.name_scope('accuracy-evaluation'):
        compared = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(compared, tf.float32))
    return accuracy


def hard_dice_evaluator(logits, labels, smooth=1e-5):
    '''Hard-dice evaluation.

    NOTE: `logits` and `labels` are all `NHWC` data.
    NOTE: labels.dtype is `int`.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor 
        ground truth.
    return: dice_score: tensorflow.Tensor
        hard dice score.
    '''
    with tf.name_scope('dice-evaluation'):
        logits = tf.cast(tf.nn.softmax(logits) + 0.5, labels.dtype)
        intersection = tf.reduce_sum(logits * labels, axis=[1,2])
        intersection = tf.cast(intersection, tf.float32)
        logits = tf.cast(tf.reduce_sum(logits, axis=[1,2]), tf.float32) 
        labels = tf.cast(tf.reduce_sum(labels, axis=[1,2]), tf.float32)

        dice_score = tf.divide(2.0 * intersection + smooth, 
            logits + labels + smooth)
        dice_score = tf.reduce_mean(dice_score, axis=[0])
        dice_score = tf.Print(dice_score, [dice_score], "hard dice score: ")
        dice_score = tf.reduce_mean(dice_score)
        
    return dice_score