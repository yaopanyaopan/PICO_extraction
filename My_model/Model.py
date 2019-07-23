# -*-coding:utf-8 -*-
__author__ = '$'
import tensorflow as tf
import numpy as np
import math
from My_model import Myattention
import tensorflow.contrib.crf

class BiLSTM_CRF(object):

    def __init__(self,
                 max_seq_len,
                 max_sent_len,
                 embedding_size,
                 hidden_units,
                 l2_reg_lambda,
                 dropout_keep_prob,
                 batch_size,
                 W_embedding,
                 num_tags,):
        self.batch_size = batch_size
        self.W_embedding = W_embedding
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.max_seq_len = max_seq_len
        self.max_sent_len = max_sent_len


        self.dropout_keep_prob = dropout_keep_prob
        self.num_tags = num_tags

        self.initializer = tf.contrib.layers.xavier_initializer()

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)


    def build(self):

        self.add_placeholders()
        self.embedding_lookup()
        self.LSTM_op()
        self.loss_op()


    def add_placeholders(self):

        # self.input_t = tf.placeholder(tf.int32, shape=[None, None], name='input_t')   # 标题

        self.input_a = tf.placeholder(tf.int32,shape=[None,None,None],name='input_a')   # 摘要

        self.labels = tf.placeholder(tf.int32,shape=[None,None], name='labels')   # 标签序列

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None] , name='sequence_lengths')   # num of sentence in a Doc

        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')


    def embedding_lookup(self):

        self.W = tf.get_variable(initializer=self.W_embedding, dtype=tf.float32, trainable=True, name="vocabulary")
        self.embedding = tf.concat([tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32)),
                                         tf.get_variable("unk", [1, self.embedding_size], dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer()),self.W], 0)

        # self.t_embeddings = tf.nn.embedding_lookup(self.embedding, self.input_t)
        self.a_embeddings = tf.nn.embedding_lookup(self.embedding, self.input_a)    # 查词表


    def LSTM_op(self):

            x1 = tf.reshape(self.a_embeddings,[-1 ,self.max_sent_len ,self.embedding_size])

            with tf.variable_scope("biLSTM_word", reuse=tf.AUTO_REUSE):
                fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
                # fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=drop_out)
                bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)

                ((fw_outputs, bw_outputs), (fw_final, bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    x1,
                    dtype=tf.float32)

                outputs_w = tf.concat((fw_outputs, bw_outputs), 2)
                last_out = tf.concat((fw_final.h, bw_final.h), 1)

                self.word_level_output, self.alphas_w = Myattention.attention(outputs_w, self.max_sent_len,self.hidden_units * 2, return_alphas=True,regularizer=self.regularizer,name="word_level")

                self.word_level_output = tf.nn.dropout(self.word_level_output, self.dropout_keep_prob)

            with tf.variable_scope("biLSTM_sentence", reuse=tf.AUTO_REUSE):

                x2 = tf.reshape(self.word_level_output,[-1, self.max_seq_len, self.hidden_units * 2])

                fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
                # fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=drop_out)
                bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)

                ((fw_outputs, bw_outputs), (fw_final, bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    x2,
                    dtype=tf.float32)

                outputs_s = tf.concat((fw_outputs, bw_outputs), 2)
                last_out = tf.concat((fw_final.h, bw_final.h), 1)


            with tf.variable_scope('proj'):
                w = tf.get_variable(name='w',shape =[ self.hidden_units * 2 , self.num_tags],initializer=self.initializer,dtype=tf.float32)
                b = tf.get_variable(name='b',shape=[self.num_tags],initializer=tf.zeros_initializer,dtype=tf.float32)

                s = tf.shape(outputs_s)

                outputs_s = tf.reshape(outputs_s, [-1, 2*self.hidden_units])

                pred = tf.matmul(outputs_s, w) + b

                self.logits = tf.reshape(pred, [-1, s[1] ,self.num_tags])

                tf.add_to_collection('logits',self.logits)


    def loss_op(self):

        log_likelihood , self.transition_params = tf.contrib.crf.crf_log_likelihood(
            inputs=self.logits, tag_indices=self.labels, sequence_lengths =self.sequence_lengths)

        tf.add_to_collection('transition_params',self.transition_params)

        self.loss = tf.reduce_mean(-log_likelihood)


        l2 = 0.0001 * sum([
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("u" in tf_var.name or "b" in tf_var.name)])
        self.loss += l2



