# -*- coding: utf-8 -*-

'''Data loss.'''

import tensorflow as tf


def cross_entropy_loss(logits, labels, loss_for='classification'):
    '''Calculate cross entropy loss mainly for classification.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor
        ground truth.
    param: loss_for: str, optional
        computing loss for which type of task.
        mainly for classification, but also for segmentation.
    return: data_loss: tensorflow.Tensor
        data loss scalar.
    '''
    with tf.name_scope('{}-cross-entropy'.format(loss_for)):
        data_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels))
        # data_loss = tf.clip_by_value(data_loss, 0.0, 10.0) # TODO
        data_loss = tf.Print(data_loss, [data_loss], "cross entropy: ")
        tf.summary.scalar('loss', data_loss)

    return data_loss


def hinge_loss(logits, labels, m1=0.9, m2=0.1, r=0.5,
        loss_for='classification'):
    '''Hinge loss for classification. This definition comes
    from "Dynamic Routing Between Capsules".

    param: logits: tensorflow.Tensor
        predicted labels.
    param: labels: tensorflow.Tensor
        ground truth.
    param: m1: tf.float32
        the lower bound for identitying right classification.
    param: m2: tf.float32
        the upper bound for identitying wrong classification.
    param: r: tf.float32
        the cofficient of 'wrong_part' to ensure the numerical stability.
    '''
    with tf.name_scope('{}-hinge-loss'.format(loss_for)):
        logits = tf.nn.softmax(logits)
        labels = tf.cast(labels, tf.float32)

        correct_part = tf.square(tf.nn.relu(m1 - logits)) * labels
        wrong_part = tf.square(tf.nn.relu(logits - m2)) * (1 - labels)
        loss = tf.reduce_mean(correct_part + r * wrong_part)
        tf.summary.scalar('loss', loss)
    
    return loss


def soft_dice(logits, labels, smooth=1e-5):
    '''Calculate soft dice in segmentation task.

    NOTE: `logits` and `labels` are all `NHWC` data.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor
        ground truth.
    return: dice_score: tensorflow.Tensor
        dice score.
    '''
    logits = tf.nn.softmax(logits)
    labels = tf.cast(labels, logits.dtype)

    intersection = tf.reduce_sum(logits * labels, axis=[1,2])
    logits = tf.reduce_sum(logits, axis=[1,2])
    labels = tf.reduce_sum(labels, axis=[1,2])
    dice_score = tf.divide(2.0 * intersection + smooth, 
        logits + labels + smooth)
    dice_score = tf.reduce_mean(dice_score, axis=[0])

    return dice_score


def dice_loss(logits, labels, smooth=1e-5, loss_for='segmentation'):
    '''Calculate dice loss for segmentation related task.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor
        ground truth.
    return: data_loss: tensorflow.Tensor
        dice loss.
    '''
    with tf.name_scope('{}-dice'.format(loss_for)):
        dice_score = soft_dice(logits, labels, smooth)
        avg_dice = tf.reduce_mean(dice_score)
        tf.summary.scalar('avg-soft-dice', avg_dice)

        loss = 1 - avg_dice
        tf.summary.scalar('loss', loss)

    return loss


def weighted_dice_loss(logits, labels, smooth=1e-5, loss_for='segmentation'):
    '''Calculate dice loss for segmentation related task.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor
        ground truth.
    return: data_loss: tensorflow.Tensor
        dice loss.
    '''
    with tf.name_scope('{}-dice'.format(loss_for)):
        dice_score = soft_dice(logits, labels, smooth)
        
        avg_dice = tf.reduce_mean(dice_score)
        tf.summary.scalar('avg-soft-dice', avg_dice)

        weights = tf.constant([[1, 100]], dtype=tf.float32)
        dice_score = weights * dice_score
        avg_dice = tf.reduce_mean(dice_score)
        loss = 100 - avg_dice
        tf.summary.scalar('loss', loss)

    return loss



def dice_hinge_loss(logits, labels, ratio=0.5, smooth=1e-5, 
                    loss_for='segmentation'):
    '''Calcalate dice and hingle loss for segmentation related task.

    param: logits: tensorflow.Tensor
        predicted lables.
    param: labels: tensorflow.Tensor
        ground truth.
    param: ratio: float
        coefficient of hinge loss.
    return: loss: tensorflow.Tensor
        loss.
    '''
    _dice_loss = dice_loss(logits, labels, smooth, loss_for)
    _hinge_loss = hinge_loss(logits, labels, loss_for=loss_for)
    with tf.name_scope("dice-hinge-loss"):
        loss = _dice_loss + ratio * _hinge_loss
        tf.summary.scalar('loss', loss)
    return loss