#!/usr/bin/python
# -*- coding: utf-8 -*-

import gzip
import logging
import os
import struct
import urllib

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t%(levelname)s\t%(filename)s:%(lineno)d\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mx_mnist_mlp")


TMP_DATA_DIR = '../tmp/data/'
MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte.gz'
TRAIN_IMAGE_FILENAME = 'train-images-idx3-ubyte.gz'
TEST_LABEL_FILENAME = 't10k-labels-idx1-ubyte.gz'
TEST_IMAGE_FILENAME = 't10k-images-idx3-ubyte.gz'


def download_data(url, dirname):
    filename = url.split("/")[-1]
    pathname = dirname + filename
    if not os.path.exists(pathname):
        logger.info("downloading %s...", filename)
        urllib.urlretrieve(url, pathname)
    else:
        logger.info("%s exists", pathname)
    return pathname


def read_data(label_url, image_url, dirname):
    with gzip.open(download_data(label_url, dirname)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url, dirname), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return label, image


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255


def main():
    logger.info("start...")

    (train_lbl, train_img) = read_data(MNIST_URL + TRAIN_LABEL_FILENAME,
                                       MNIST_URL + TRAIN_IMAGE_FILENAME,
                                       TMP_DATA_DIR)
    (val_lbl, val_img) = read_data(MNIST_URL + TEST_LABEL_FILENAME,
                                   MNIST_URL + TEST_IMAGE_FILENAME,
                                   TMP_DATA_DIR)

    logger.info("data is ok")

    # plot 10 images
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(train_img[i], cmap='Greys_r')
        plt.axis('off')
    print 'label: %s' % train_lbl[0: 10]
    print 'close the plot to continue'
    plt.show()

    batch_size = 100

    # create data iterators
    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

    logger.info("define the network...")

    # Create a place holder variable for the input data
    data = mx.sym.Variable('data')
    # Flatten the data from 4-D shape (batch_size, num_channel, width, height)
    # into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    # Apply relu to the output of the first fully-connected layer
    act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

    # The second fully-connected layer and the according activation function
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

    # The third fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
    # The softmax and loss layer
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    logger.info("model is ok")

    # We visualize the network structure with output size (the batch_size is ignored.)
    shape = {"data": (batch_size, 1, 28, 28)}
    name = "network_structure"
    dot = mx.viz.plot_network(symbol=mlp, shape=shape)
    dot.render(name)
    print "%s.pdf is created" % name

    model = mx.model.FeedForward(
        symbol=mlp,        # network structure
        ctx=mx.gpu(0),     # gpu/cpu
        num_epoch=10,      # number of data passes for training
        learning_rate=0.1  # learning rate of SGD
    )

    model.fit(
        X=train_iter,        # training data
        eval_data=val_iter,  # validation data
        batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
    )

    # give a prediction example
    plt.imshow(val_img[0], cmap='Greys_r')
    plt.axis('off')
    prob = model.predict(val_img[0: 1].astype(np.float32) / 255)[0]
    print 'Classified as %d with probability %f' % (prob.argmax(), max(prob))
    print 'close the plot to continue'
    plt.show()

    print 'Validation accuracy: %f%%' % (model.score(val_iter) * 100)


if __name__ == '__main__':
    main()
