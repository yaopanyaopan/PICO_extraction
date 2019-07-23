# -*- coding: utf-8 -*
__author__ = '$'

import sys
import tensorflow as tf
import numpy as np
import math
from My_model import input_helpers
# from My_model import input_helpers
import os
import re
import time
import datetime
from Model_acl import Model
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.flags.DEFINE_integer("embedding_dim",300,"...")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"...")
tf.flags.DEFINE_float("l2_reg_lambda",0.0001,"...")
tf.flags.DEFINE_integer("hidden_units",150,"...")
tf.flags.DEFINE_integer("batch_size",40,"...")
tf.flags.DEFINE_integer("num_epochs",40,"...")
tf.flags.DEFINE_integer("num_tags",7,"...")

tf.flags.DEFINE_integer("evaluate_every",1,"...")    # 验证集上评估
tf.flags.DEFINE_integer("checkpoint_every",1,"...")          # 保存节点步数

Flags = tf.flags.FLAGS

print("参数:\n")
for attr,value in Flags.flag_values_dict().items():
    print("{} = {}".format(attr.upper(), value))

inpH = input_helpers.InputHelper

train,dev,label_train,label_dev,train_length,dev_length, max_sentence_size,max_word_size , W_embedding = input_helpers.getDataSets()

print("初始化模型")
with tf.Graph().as_default():

    gpuconfig = tf.ConfigProto()
    # gpuconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=gpuconfig)
    with sess.as_default():

        BCModel = Model.BiLSTM_CRF(
            max_seq_len=max_sentence_size,
            max_sent_len=max_word_size,
            embedding_size=Flags.embedding_dim,
            hidden_units=Flags.hidden_units,
            l2_reg_lambda=Flags.l2_reg_lambda,
            dropout_keep_prob=Flags.dropout_keep_prob,
            batch_size=Flags.batch_size,
            W_embedding=W_embedding,
            num_tags = Flags.num_tags)

        BCModel.build()
        global_step = tf.Variable(0,name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)

        # grads_and_vars = optimizer.compute_gradients(BCModel.loss)
        # tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grads, vs = zip(*optimizer.compute_gradients(BCModel.loss))
        grads, gnorm = tf.clip_by_global_norm(grads,10)
        tr_op_set = optimizer.apply_gradients(zip(grads, vs), global_step=global_step)
        # 记录梯度值
        grad_summaries = []
        # for g,v in grads_and_vars:
        for g,v in zip(grads,vs):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)
        timestap = str(time.time())
        out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestap))

        loss_summary = tf.summary.scalar("loss", BCModel.loss)   # 记录损失值
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir,"summaries","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())
        print("初始化所有变量")

        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open(os.path.join(checkpoint_dir, "graphpb.txt"), "w") as f:
            f.write(graphpb_txt)

        def train_step(input_a, labels ,sequence_length ,epoch):

            feed_dict = {
                # Model.input_t: input_t,
                BCModel.input_a: input_a,
                BCModel.labels: labels,
                BCModel.sequence_lengths:sequence_length,
                BCModel.dropout_keep_prob: Flags.dropout_keep_prob,
            }

            _,step, loss, logits, transition_params = sess.run(
                [tr_op_set, global_step, BCModel.loss, BCModel.logits, BCModel.transition_params],feed_dict)

            print("epoch: %i , step: %i  , loss:  %f" %(epoch ,step, loss ))
            summary_op_out = sess.run(train_summary_op, feed_dict=feed_dict)
            train_summary_writer.add_summary(summary_op_out, step)

            viterbi_sequences = []
            for logit, document_length, label in zip(logits, sequence_length, labels):
                # print(logit)
                # print(document_length)
                logit = logit[:document_length]

                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)

                viterbi_sequences += [viterbi_seq]

            # print(viterbi_sequences)
            # print(sequence_length)
            return viterbi_sequences, sequence_length, labels, loss


        def dev_step(input_a, labels ,sequence_length ):

            feed_dict = {
                # Model.input_t: input_t,
                BCModel.input_a: input_a,
                BCModel.labels: labels,
                BCModel.sequence_lengths:sequence_length,
                BCModel.dropout_keep_prob: 1.0,
            }

            loss,logits, transition_params = sess.run(
                [BCModel.loss,BCModel.logits, BCModel.transition_params],feed_dict)
            viterbi_sequences = []
            for logit,document_length,label in zip(logits,sequence_length,labels):

                logit = logit[:document_length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                viterbi_sequences += [viterbi_seq]
            return viterbi_sequences, sequence_length,labels,loss,transition_params


        ptr = 0
        max_validation_acc = 0.0
        print("开始训练")
        epoch = []
        train_loss = []
        train_acc = []
        dev_loss = []
        dev_acc = []

        for ep in range(0,int(Flags.num_epochs)):
            epoch.append(ep)
            if ep % Flags.evaluate_every == 0:
                print("评估：")
                dev_batches = inpH.batch_iter1(list(zip(dev, label_dev,dev_length)), Flags.batch_size, False)

                VS = []
                SL = []
                L = []
                loss_avg = []
                for db in dev_batches:
                    if len(db) < 1:
                        continue

                    input_a, labels, sequence_length = zip(*db)
                    viterbi_sequences, sequence_length, labels, loss,trans_param = dev_step(input_a, labels, sequence_length)

                    VS += viterbi_sequences
                    SL += sequence_length
                    L += labels
                    loss_avg.append(loss)

                acc = input_helpers.count_acc(VS,SL,L)
                dev_acc.append(acc)
                dev_loss.append(sum(loss_avg)/len(loss_avg))
                print('\n')
                print("----DEV:-----Loss: %f--------acc: %f" % (sum(loss_avg)/len(loss_avg),acc))
                print(trans_param)
                sum_acc = acc

            if ep % Flags.checkpoint_every == 0 :
                # if sum_acc >= max_validation_acc:  # 保存在验证集准确率最大的模型
                    max_validation_acc = sum_acc
                    current_step = tf.train.global_step(sess, global_step)
                    saver.save(sess, checkpoint_dir, global_step=current_step)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix,
                                         "graph" + str(time.time()) + ".pb", as_text=False)
            batches = inpH.batch_iter(
                list(zip(train,label_train,train_length)),
                Flags.batch_size,
                 True)
            VS = []
            SL = []
            L = []
            sum_loss = []
            for batch in batches:
                if len(batch) < 1:
                    continue
                input_a, labels, sequence_length = zip(*batch)

                viterbi_sequences, sequence_length, labels,loss = train_step(input_a, labels,sequence_length,ep)    # 训练
                VS += viterbi_sequences
                SL += sequence_length
                L += labels
                sum_loss.append(loss)

            acc = input_helpers.count_acc(VS, SL, L)
            train_acc.append(acc)
            train_loss.append(sum(sum_loss)/len(sum_loss))
            print('\n')
            print('epoch : %i , average-loss : %f , acc: %f '%(ep,sum(sum_loss)/len(sum_loss),acc))
            print('\n')

        # 画图
        def to_picture(title, x_content, y_content, xlabel, ylabel, xlim, ylim, path):
            print("    - [Info] Plotting metrics into picture " + path)
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = True
            plt.figure(figsize=(10, 5))
            plt.grid(linestyle="--")
            plt.xlim(xlim)
            plt.ylim(ylim)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.plot(x_content, y_content)
            plt.xlabel(xlabel, fontsize=13, fontweight='bold')
            plt.ylabel(ylabel, fontsize=13, fontweight='bold')
            plt.savefig(path, format='png')
            plt.clf()

        out_dir = os.path.join(out_dir)
        to_picture(title='dev-loss', x_content=epoch, y_content=dev_loss, xlabel='Epoch', ylabel='loss', xlim=(0,40) ,ylim=(0,40), path=out_dir+'/dev-loss.png')
        to_picture(title='dev-acc', x_content=epoch, y_content=dev_acc, xlabel='Epoch', ylabel='acc', xlim=(0, 40),ylim=(0,1),
                   path=out_dir + '/dev-acc.png')
        to_picture(title='train-loss', x_content=epoch, y_content=train_loss, xlabel='Epoch', ylabel='loss', xlim=(0, 40),ylim=(0,40),
                   path=out_dir + '/train-loss.png')
        to_picture(title='train-acc', x_content=epoch, y_content=train_acc, xlabel='Epoch', ylabel='acc', xlim=(0, 40),
                   ylim=(0, 1),
                   path=out_dir + '/train-acc.png')






