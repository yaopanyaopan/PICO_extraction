# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import csv
import pickle
import sys
import tensorflow as tf
import numpy as np

from My_model import input_helpers
import os


FLAGS = tf.flags.FLAGS

print('\n Parameters:')
for attr, value in sorted(FLAGS._flags().items()):
    print('{}={}'.format(attr.upper(),value))


inpH = input_helpers.InputHelper

pkl1 = open('../data/test.pickle','rb')
test = pickle.load(pkl1)
pkl2 = open('../data/label_test.pickle','rb')
label_test = pickle.load(pkl2)
pkl3 = open('../data/test_length.pickle','rb')
test_length = pickle.load(pkl3)

#----------------------评 估----------------------------------------------------------
print('\nEvaluating...\n')

checkpoint_file = tf.train.latest_checkpoint('runs/1558539232.3026383')

graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess,checkpoint_file)

        input_x = graph.get_operation_by_name('input_a').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        sequence_lengths = graph.get_operation_by_name('sequence_lengths').outputs[0]

        logits = tf.get_collection('logits')[0]
        transition_params = tf.get_collection('transition_params')[0]

        batches = inpH.batch_iter1(list(zip(test, label_test, test_length)),256, False)

        VS = []
        SL = []
        L = []
        sum_loss = []
        for db in batches:

            T,LT,TL = zip(*db)

            pre_logits , pre_transition_params = sess.run(
                [logits, transition_params],{input_x:T ,dropout_keep_prob:1.0 ,sequence_lengths:TL}
            )

            viterbi_sequences = []

            for logit,sequence_length in zip(pre_logits,TL):

                logit = logit[:sequence_length]

                # 返回最好的标签序列
                # print(pre_transition_params)
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, pre_transition_params)

                viterbi_sequences += [viterbi_seq]

            VS += viterbi_sequences
            SL += TL
            L += LT


        acc = input_helpers.count_acc(VS, SL, L)

        print('准确率：',acc)
        print(pre_transition_params)