import argparse
import logging
import os

import data_helpers
import mxnet as mx
import numpy as np

logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------
# argparse
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="lstm text classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained-embedding', type=bool, default=False,
                    help='use pre-trained word2vec')
parser.add_argument('--embed-size', type=int, default=300,
                    help='embedding layer size')
parser.add_argument('--gpus', type=str, default='0',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--kv-store', type=str, default='local',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='max num of epochs')
parser.add_argument('--batch-size', type=int, default=64,
                    help='the batch size')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--num-hidden', type=int, default=50,
                    help='num of hidden units')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--save-period', type=int, default=1,
                    help='saves a model checkpoint every n epochs')
args = parser.parse_args()


def save_model(prefix, dir="../checkpoint"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return mx.callback.do_checkpoint(prefix=os.path.join(dir, prefix), period=args.save_period)


def data_iter(batch_size, embed_size=300, pre_trained_word2vec=False):
    print('Loading data...')
    if pre_trained_word2vec:
        word2vec = data_helpers.load_pretrained_word2vec('data/rt.vec')
        x, y = data_helpers.load_data_with_word2vec(word2vec)
        # reshpae for convolution input
        x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        embed_size = x.shape[-1]
        sentence_size = x.shape[2]
        vocab_size = -1
    else:
        x, y, vocab, vocab_inv = data_helpers.load_data()
        embed_size = embed_size
        sentence_size = x.shape[1]
        vocab_size = len(vocab)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/valid set
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
    print('Train/Valid split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('valid shape:', x_dev.shape)
    print('sentence max words', sentence_size)
    print('embedding size', embed_size)
    print('vocab size', vocab_size)

    train = mx.io.NDArrayIter(x_train, y_train, batch_size, shuffle=True)
    valid = mx.io.NDArrayIter(x_dev, y_dev, batch_size)

    return (train, valid, sentence_size, embed_size, vocab_size)


def sym_gen(batch_size, sentence_size, embed_size, vocab_size,
            num_classes=2, num_hidden=100, pre_trained_word2vec=False):

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    if not pre_trained_word2vec:
        word_vec = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=embed_size, name='word_embedding')
    else:
        word_vec = data

    lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)
    lstm_cell.reset()
    outputs, states = lstm_cell.unroll(length=sentence_size,
                                       inputs=word_vec,
                                       layout='NTC',
                                       merge_outputs=True)

    flatten = mx.sym.reshape(data=outputs, shape=(batch_size, sentence_size * num_hidden))
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    output = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')

    return output


def train(symbol, train_iter, valid_iter):
    devs = mx.cpu() if args.gpus is None or args.gpus == '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    mod = mx.mod.Module(symbol, context=devs)
    mod.fit(train_data           = train_iter,
            eval_data            = valid_iter,
            eval_metric          = 'accuracy',
            kvstore              = args.kv_store,
            optimizer            = args.optimizer,
            optimizer_params     = {'learning_rate': args.lr},
            initializer          = mx.initializer.Uniform(0.1),  # Uniform(0.1)-->0.79 Normal(0.01)-->0.75
            num_epoch            = args.num_epochs,
            batch_end_callback   = [mx.callback.Speedometer(args.batch_size, frequent=args.disp_batches)],
            epoch_end_callback   = save_model('lstm'))


if __name__ == '__main__':
    # data iter
    train_iter, valid_iter, sentence_size, embed_size, vocab_size = data_iter(args.batch_size,
                                                                              args.embed_size,
                                                                              args.pretrained_embedding)

    # network symbol
    symbol = sym_gen(args.batch_size,
                     sentence_size,
                     embed_size,
                     vocab_size,
                     num_classes=2,
                     num_hidden=args.num_hidden,
                     pre_trained_word2vec=args.pretrained_embedding)

    # train cnn model
    train(symbol, train_iter, valid_iter)
