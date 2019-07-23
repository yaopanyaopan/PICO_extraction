# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np

def attention(inputs, attention_size, hidden_units,return_alphas,regularizer,name):

    with tf.variable_scope("attention"+name,reuse=tf.AUTO_REUSE):


        hidden_size = hidden_units     # LSTM隐藏层大小 是双向拼接后的

        # 注意力机制
        W_omega = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer,shape=[hidden_size , attention_size],dtype=tf.float32,name=name+"w")
        b_omega = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[attention_size],dtype=tf.float32,name=name+"b")
        u_omega = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[attention_size],dtype=tf.float32,name=name+"u")

        v = tf.tanh(tf.matmul(tf.reshape(inputs,[-1,hidden_size]),W_omega) + tf.reshape(b_omega,[1,-1]))
        # print("v",v)

        vu = tf.matmul(v , tf.reshape(u_omega,[-1, 1]))
        # print("vu",vu)
        exps = tf.reshape(tf.exp(vu) ,[-1,attention_size])
        # print("exps",exps)
        alphas = exps / tf.reshape(tf.reduce_sum(exps,1), [-1,1])

        # print("inputs",inputs)
        # print("alphas",alphas)

        # print("ATT",inputs)
        # print("alphas",alphas)
        output = tf.reduce_sum(inputs * tf.reshape(alphas,[-1,attention_size,1]), 1)

        # print("output",output)

        return output , alphas







