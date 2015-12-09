import numpy
import time
import sys
import subprocess
import os
import random

from action.data.load import *
from action.rnn.elman import model
# from is13.metrics.accuracy import conlleval
from action.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.05,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':190, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':10, # dimension of word embedding
         'nepochs':50}

    folder = "C:/SciSoft/rnn-exp/action/elman-forward"

    # load the dataset
    # dm = data_manager()
    # dm.load_all("F:/data/S1_T")
    # # train_lex, train_y = dm.gen_set(s['win'])
    # train_lex, train_y = dm.gen_set_full()

    dm = data_manager()
    dm.load_all("F:/data/S2")
    # test_lex, test_y = dm.gen_set(s['win'])
    test_lex, test_y = dm.gen_set_full()
    test_lex_flatten = []
    test_y_flatten = []
    for i in xrange(len(test_lex)):
        test_lex_flatten += test_lex[i]
        test_y_flatten += test_y[i]

    # train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    # idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    # idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    # train_lex, train_ne, train_y = train_set
    # valid_lex, valid_ne, valid_y = valid_set
    # test_lex,  test_ne,  test_y  = test_set

    vocsize = 190 # distances
    # vocsize = 190 + 60 # distances + coordinates
    nclasses = 7
    # nframes = len(train_lex)

    # # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )
    rnn.load(folder)

    predictions_test = [ int(rnn.classify([x])) for x in test_lex_flatten ]
    groundtruth_test = [ y for y in test_y_flatten ]
    print predictions_test
    # print groundtruth_test
    stats = [0]*7
    error_count = 0
    for i in xrange(len(predictions_test)):
        if predictions_test[i] != groundtruth_test[i]:
            error_count += 1
        stats[predictions_test[i]] += 1
    accuracy =  1 - float(error_count)/len(predictions_test)
    print accuracy
    print [float(x)/len(predictions_test) for x in stats]
    #     words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

    #     predictions_valid = [ map(lambda x: idx2label[x], \
    #                          rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
    #                          for x in valid_lex ]
    #     groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
    #     words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

    #     # evaluation // compute the accuracy using conlleval.pl
    #     res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
    #     res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

    #     if res_valid['f1'] > best_f1:
    #         rnn.save(folder)
    #         best_f1 = res_valid['f1']
    #         if s['verbose']:
    #             print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
    #         s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
    #         s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
    #         s['be'] = e
    #         subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
    #         subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
    #     else:
    #         print ''

    #     # learning rate decay if no improvement in 10 epochs
    #     if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
    #     if s['clr'] < 1e-5: break

    # print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

