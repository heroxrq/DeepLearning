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
logger = logging.getLogger("tf_mnist_softmax")

FLAGS = None


def main():
    logger.info("start...")

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    logger.info("data is ok")

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    logger.info("model is ok")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    logger.info("loss is ok")

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    logger.info("optimizer is ok")

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    logger.info("start training...")
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    logger.info("start evaluating...")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../tmp/data', help='Directory for storing data')
    FLAGS = parser.parse_args()
    main()
