# Copyright 2017 Phonexia (author: Josef Slavicek)
# Licensed under the Apache License, Version 2.0 (the "License")

# Simple neural network - LRE classifier using i-vectors as input. Hidden layers of the net are Variational information
# bottlenecks (VIB) [Alemi2016DeepVI https://arxiv.org/pdf/1612.00410.pdf]. The implementation allows for more hidden
# layers but our experiments suggests that one such bottleneck is enough :-)

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adam, Optimizer
import argparse
import gzip
import keras
import keras.backend as K
import keras.initializers
import numpy as numpy
import os
import pickle
import shutil
import sys


class DataLoader(object):
    """
    Class responsible for loading data into memory from 'somewhere' (e.g. disk)
    """

    @staticmethod
    def get_train_data_x():
        # type: () -> numpy.ndarray
        """
        Return training dataset input (i-vectors) as numpy matrix.
        :return: (numpy.ndarray) matrix containing input ivectors (vectors in rows).
        """
        raise Exception('TODO - implement this method')

    @staticmethod
    def get_train_data_t():
        # type: () -> numpy.ndarray
        """
        Return training dataset target vectors (one-hot encoded target classes) as numpy matrix.
        :return: (numpy.ndarray) matrix containing one-hot encoded target vectors (vectors in rows).
        """
        raise Exception('TODO - implement this method')

    @staticmethod
    def get_dev_data_x():
        # type: () -> numpy.ndarray
        """
        Return validation dataset input (i-vectors) as numpy matrix.
        :return: (numpy.ndarray) matrix containing input ivectors (vectors in rows).
        """
        raise Exception('TODO - implement this method')

    @staticmethod
    def get_dev_data_t():
        # type: () -> numpy.ndarray
        """
        Return validation dataset target vectors (one-hot encoded target classes) as numpy matrix.
        :return: (numpy.ndarray) matrix containing one-hot encoded target vectors (vectors in rows).
        """
        raise Exception('TODO - implement this method')


class Settings(object):
    """
    Hardcoded values of some training (hyper)parameters
    """
    #: number of hidden layers
    h_n_layers = 1
    #: number of units in single hidden layer (K in [Alemi2016DeepVI])
    h_dim = 64
    #: learning rate
    lr = 3.0e-5            # result of tuning to learn in 512 epochs - higher values of lr works too and converge faster
    #: L2 regularization coefficient
    l2 = 1.0e-5
    #: dropout applied to input layer
    in_dropout = 7.0e-2
    #: relative weight of KL cost term (beta in [Alemi2016DeepVI])
    kl_w = 2.0e-2
    #: number of training epochs
    n_epoch = 512
    #: size of training minibatch
    batch_size = 64
    #: size of training epoch (because of balanced DataSource, 'epoch' has not natural meaning)
    epoch_size = 16 * 4096


class FakeOptimizer(Optimizer):
    """
    Keras optimizer which does nothing - it does not change any value in NN
    """

    def __init__(self, **kwargs):
        super(FakeOptimizer, self).__init__(**kwargs)

    def get_updates(self, params, constraints, loss):
        return list()

    def get_config(self):
        return super(FakeOptimizer, self).get_config()


class CLSFNN(object):
    """
    NN for LRE classification: i-vectors as inputs, class posteriors as outputs
    """

    def __init__(self, in_dim, t_dim, h_n_layers, h_dim, in_dropout, kl_w, l2, lr):
        # type: (int, int, int, int, float, float, float, float) -> None
        """
        constructor :-)
        :param in_dim: (int) dimensionality of input (i-vectors) [1 ... inf]
        :param t_dim: (int) dimensionality of target - number of classes [2 ... inf]
        :param h_n_layers: (int) number of hidden layers [1 ... inf]
        :param h_dim: (int) dimensionality of hidden layers [1 ... inf]
        :param in_dropout: (float) dropout applied to input [0.0 ... 1.0]
        :param kl_w: (float) relative weight of KL cost (beta in [Alemi2016DeepVI]) [0.0 ... 1.0]
        :param l2: (float) L2 regularizer applied to all layers [0.0 ... inf]
        :param lr: (float) learnign rate [0.0 ... inf]
        """

        in_x = Input(shape=(in_dim,))               # input - ivectors (matrix with vectors in lines)
        val_drp = Dropout(in_dropout)(in_x)
        # from now we are building two models in parallel with some layers shared:
        val_fit = val_drp       # *_fit is for training
        val_prd = val_drp       # *_prd for testing (prediction)
        # loss_val will sum value of KL loss accross all layers
        val_kl_loss = Lambda(lambda x: 0.0 * K.sum(x, axis=1), output_shape=(1,))(val_fit)   # crazy way to get 0 vector
        for _ in xrange(h_n_layers):        # building hidden layers
            denselayer = Dense(h_dim, activation='linear', kernel_regularizer=keras.regularizers.l2(l2))
            val_fit = denselayer(val_fit)
            val_prd = denselayer(val_prd)
            lrelulayer = LeakyReLU(0.1)
            val_fit = lrelulayer(val_fit)
            val_prd = lrelulayer(val_prd)
            mulayer = Dense(h_dim, activation='linear', kernel_regularizer=keras.regularizers.l2(l2))
            val_mu = mulayer(val_fit)
            val_log_sigma = Dense(h_dim, activation='linear', kernel_regularizer=keras.regularizers.l2(l2))(val_fit)
            val_kl_loss = Lambda(CLSFNN.__accum_kl_loss, output_shape=(1,))([val_mu, val_log_sigma, val_kl_loss])
            val_fit = Lambda(CLSFNN.__sample_z, output_shape=(h_dim,))([val_mu, val_log_sigma])
            val_prd = mulayer(val_prd)
        # final softmax layer
        dense = Dense(t_dim, activation='softmax', kernel_regularizer=keras.regularizers.l2(l2))
        val_fit = dense(val_fit)
        val_prd = dense(val_prd)
        # model for training - first output is standard softmax predictor, second output is KL loss of hidden layers
        self.model_fit = Model(inputs=[in_x], outputs=[val_fit, val_kl_loss])
        self.model_fit.compile(optimizer=Adam(lr=lr), loss=['categorical_crossentropy', lambda t, y: y],
                               loss_weights=[1.0 - kl_w, kl_w])
        # model for testing
        self.model_prd = Model(inputs=[in_x], outputs=[val_prd])
        self.model_prd.compile(optimizer=FakeOptimizer(), loss=lambda t, y: 0.0 * K.sum(y))

    @staticmethod
    def __accum_kl_loss(args):
        """
        Accumulator of KL loss
        :param args: (list) list of arguments: mu, log_sigma, old_loss. mu and log_sigma are outputs from Variational
            information bottleneck (VIB) layer (parameters of sampling). old_loss is the value of KL loss accumulated on
            previous VIB layers
        :return: old_loss + KL_loss_of_mu_and_log_sigma
        """
        pmu, plog_sigma, p_old_loss = args
        kl = 0.5 * K.sum(K.exp(plog_sigma) + K.square(pmu) - 1. - plog_sigma, axis=1)
        return kl + p_old_loss

    @staticmethod
    def __sample_z(args):
        """
        Draw sample according to mu, log_sigma returned from Variational information bottleneck (VIB) layer
        :param args: (list) list of arguments: mu, log_sigma. mu and log_sigma are outputs from VIB
            layer (parameters of sampling).
        :return: random sample vector drawn from distribution specified by mu, log_sigma
        """
        mu, log_sigma = args
        eps = K.random_normal(shape=mu.shape, mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps


class InMemoryDataSource(object):
    """
    Class used to provide data (i-vectors + target) to learning method. Implements class balancing.
    """
    def __init__(self, x, t, balanced):
        # type: (numpy.ndarray, numpy.ndarray) -> None
        """
        Constructor
        :param x: (numpy.ndarray) input - ivectors, vectors in rows
        :param t: (numpy.ndarray) target - one hot encoded vectors in rows
        :param balanced: (bool) if True, the data returned from get_batch() will be class balanced
        """
        #: if True, the data returned from get_batch will be class balanced
        self.balanced = balanced        # type: bool
        #: input - ivectors, vectors in rows
        self.x = x                      # type: numpy.ndarray
        #: target - one hot encoded vectors in rows
        self.t = t                      # type: numpy.ndarray
        #: list, contains tuples (sub_x, sub_t), where i-th tuple contains part of x and t belonging to i-th class
        self.bins = list()              # type: list
        for i in xrange(t.shape[1]):
            xbin = x[t[:, i] == 1, :]
            tbin = t[t[:, i] == 1, :]
            self.bins.append((xbin, tbin))

    def get_batch(self, batch_size):
        # type: (int) -> (numpy.ndarray, numpy.ndarray)
        """
        Returns data batch of required size
        :param batch_size: (int) required size of batch - number of datapoints
        :return: (tuple) batch as a tuple containing x and t where x is matrix containing ivectors in rows and t is
            matrix containing one-hot encoded targets in rows.
        """
        if not self.balanced:
            if self.x.shape[0] <= batch_size:
                return self.x, self.t
            else:
                idxs = numpy.random.randint(self.x.shape[0], size=batch_size)
                return self.x[idxs, :], self.t[idxs, :]
        else:
            cnt = batch_size / len(self.bins)
            xs = list()
            ts = list()
            for xbin, tbin in self.bins:
                idxs = numpy.random.randint(xbin.shape[0], size=cnt)
                xs.append(xbin[idxs, :])
                ts.append(tbin[idxs, :])
            x = numpy.concatenate(xs, axis=0)
            t = numpy.concatenate(ts, axis=0)
            idxs = numpy.arange(x.shape[0], dtype=numpy.int32)
            numpy.random.shuffle(idxs)
            return x[idxs, :], t[idxs, :]


class ScoresAndWeights(object):
    """
    Auxiliary object - keeps track of best achieved scores and corresponding NN weights
    """
    def __init__(self):
        # type: () -> None
        #: dict mapping dataset labels (str) to best achieved score (float)
        self.best_scores = dict()
        #: dict mapping dataset labels (str) to model weights (list) corresponding to best achieved score
        self.best_weights = dict()

    def record_score(self, score, label, model, print_best_marker):
        # type: (float, str, keras.models.Model, bool) -> None
        """
        Record achieved score into internal data structures, print message to stdout
        :param score: (float) achieved score where higher number means better score (e.g. accuracy)
        :param label: (str) label of the dataset. Scores for different datasets are kept independently.
        :param model: (keras.models.Model) keras model which achieved the score
        :param print_best_marker: (bool) if True, some marker will be print if current score is the best
        """
        is_best = label not in self.best_scores or self.best_scores[label] < score
        print('score_%s: %f%s' % (label, score, ' (new best)' if (is_best and print_best_marker) else ''))
        if is_best:
            self.best_scores[label] = score
            self.best_weights[label] = model.get_weights()
        sys.stdout.flush()


def score_model(model, x, t):
    # type: (keras.models.Model, numpy.ndarray, numpy.ndarray) -> float
    """
    Return score (accuracy) of given model on given data
    :param model: (keras.models.Model) model which will be scored
    :param x: (numpy.ndarray) matrix containing i-vectors in rows
    :param t: (numpy.ndarray) matrix containing one-hot encoded targets in rows
    :return: (float) accuracy - fraction of correctly classified datapoints out of all datapoints
    """
    y = model.predict(x)
    y_cls = numpy.argmax(y, axis=1)
    t_cls = numpy.argmax(t, axis=1)
    score = numpy.sum((y_cls == t_cls)[:]) * 1.0 / y.shape[0]       # type: float
    return score


def main():
    """
    Main method used to train the classifier
    """
    print('loading data')
    x_trn = DataLoader.get_train_data_x()
    t_trn = DataLoader.get_train_data_t()
    x_dev = DataLoader.get_dev_data_x()
    t_dev = DataLoader.get_dev_data_t()
    assert x_trn.shape[0] == t_trn.shape[0]
    assert x_dev.shape[0] == t_dev.shape[0]
    assert x_trn.shape[1] == x_dev.shape[1]
    assert t_trn.shape[1] == t_dev.shape[1]
    ds_trn = InMemoryDataSource(x_trn, t_trn, balanced=True)
    print('instantiating classifier')
    clsfnn = CLSFNN(in_dim=x_trn.shape[1], t_dim=t_trn.shape[1], h_n_layers=Settings.h_n_layers, h_dim=Settings.h_dim,
                    lr=Settings.lr, l2=Settings.l2, in_dropout=Settings.in_dropout, kl_w=Settings.kl_w)
    scores_and_weights = ScoresAndWeights()
    print('training classifier')
    for i in xrange(Settings.n_epoch):
        print('epoch %d' % i)
        x, t = ds_trn.get_batch(Settings.epoch_size)
        t_fake = numpy.zeros((x.shape[0], 1), dtype='float32')
        clsfnn.model_fit.fit([x], [t, t_fake], batch_size=Settings.batch_size, epochs=1, shuffle=True, verbose=0)
        for x, t, label in zip([x_trn, x_dev], [t_trn, t_dev], ['trn', 'dev']):
            score = score_model(clsfnn.model_prd, x, t)
            scores_and_weights.record_score(score, label, clsfnn.model_prd, label == 'dev')
    print('finished ... all the work disappeared because this script doesn\'t save weights')

if __name__ == '__main__':
    main()
