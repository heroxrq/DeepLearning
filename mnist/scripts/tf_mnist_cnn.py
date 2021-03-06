#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import logging

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t%(levelname)s\t%(filename)s:%(lineno)d\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("tf_mnist_cnn")

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    logger.info("start...")

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    logger.info("data is ok")

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    logger.info("first layer is ok")

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    logger.info("second layer is ok")

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    logger.info("fully-connected layer is ok")

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    logger.info("dropout is ok")

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    logger.info("output layer is ok")

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    logger.info("loss is ok")

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    logger.info("optimizer is ok")

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    logger.info("start training...")
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            logger.info("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    logger.info("start evaluating...")
    print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../tmp/data', help='Directory for storing data')
    FLAGS = parser.parse_args()
    main()
