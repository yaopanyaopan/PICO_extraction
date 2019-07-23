# -*- coding: utf-8 -*
import numpy as np
import re
import itertools
from collections import Counter
import time
import gc
# from tensorflow.contrib import learn
import gensim
from gensim.models.word2vec import Word2Vec
import gzip
import random
import sys, os
import jieba
import csv
import tensorflow as tf
import pickle
from collections import defaultdict
import math
# import data

from sklearn.metrics import classification_report

class InputHelper(object):

    def batch_iter(data,batch_size, only_part):
        """
        Generates a batch iterator for a dataset.
        """
        shuffle=True
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        print('随机采样 :',only_part,' -----1个epoch 的 step数-----:',num_batches_per_epoch)

        shuffled_data = []
        if shuffle==True:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                if only_part==True:
                    for shuffle_indice in shuffle_indices[:int(data_size)]:  # 每个epoch只随机取一部分数据
                        shuffled_data.append(data[shuffle_indice])
                else:
                    for shuffle_indice in shuffle_indices:
                        shuffled_data.append(data[shuffle_indice])

        for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                yield shuffled_data[start_index:end_index]

    def batch_iter1(data,batch_size, only_part):
        """
        Generates a batch iterator for a dataset.
        """
        shuffle=False
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        # print('随机采样 :',only_part,' -----1个epoch 的 step数-----:',num_batches_per_epoch)

        shuffled_data = []
        if shuffle==True:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                if only_part==True:
                    for shuffle_indice in shuffle_indices[:int(data_size)]:  # 每个epoch只随机取一部分数据
                        shuffled_data.append(data[shuffle_indice])
                else:
                    for shuffle_indice in shuffle_indices:
                        shuffled_data.append(data[shuffle_indice])
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                yield shuffled_data[start_index:end_index]

def count_acc(VS,SL,L):

    total = 0
    ACC = 0
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0

    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0
    total6 = 0


    labs = []
    labs_pred = []
    for vs,sl,l in zip(VS,SL,L):
        # print(vs)
        # print(l)
        # print(l[:sl])
        for a,b in zip(vs,l[:sl]):

            total +=1
            labs_pred.append(a)
            labs.append(b)

            if b == 0:
                total0 += 1
            elif b == 1:
                total1 += 1
            elif b == 2:
                total2 += 1
            elif b == 3:
                total3 += 1
            elif b == 4:
                total4 += 1
            elif b == 5:
                total5 += 1
            elif b == 6:
                total6 += 1


            if a==b:
                ACC +=1
                if b==0:
                    t0 +=1
                elif b==1:
                    t1 +=1
                elif b==2:
                    t2 +=1
                elif b==3:
                    t3 +=1
                elif b==4:
                    t4 +=1
                elif b==5:
                    t5 +=1
                elif b==6:
                    t6 +=1

    print(classification_report(labs, labs_pred, target_names=['R','M','O','P','I','C','A'], digits=4))
    # print('R',total0,float(t0 / total0))
    # print('M',total1,float(t1 / total1))
    # print('O',total2,float(t2 / total2))
    # print('P',total3,float(t3 / total3))
    # print('I',total4,float(t4 / total4))
    # print('C',total5,float(t5 / total5))
    # print('A',total6,float(t6 / total6))
    # print('pad',total7)
    #
    # print(total)
    return float(ACC)/total


def pre_process():

    jieba.enable_parallel(16)
    with open('../data/medical.csv','r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            jieba.add_word(i[0])

    word_freq = defaultdict(int)


    with open("../data/PICO_train.txt","r",encoding="utf8") as fr:

        reader = fr.readlines()
        for i in reader:
            ss = i.split('|')
            # print(ss)
            if len(ss)>1:
                for j in ss[-1].split():
                    word_freq[j] +=1
    with open("../data/PICO_dev.txt", "r", encoding="utf8") as fr:

        reader = fr.readlines()
        for i in reader:
            ss = i.split('|')
            # print(ss)
            if len(ss) > 1:
                for j in ss[-1].split():
                    word_freq[j] += 1

    with open("../data/PICO_test.txt", "r", encoding="utf8") as fr:

        reader = fr.readlines()
        for i in reader:
            ss = i.split('|')
            # print(ss)
            if len(ss) > 1:
                for j in ss[-1].split():
                    word_freq[j] += 1

    with open("../data/word_freq.csv","w",encoding='utf8') as fw:
        res = []
        for k,v in word_freq.items():
            res.append([k,v])

        writer_res = csv.writer(fw)
        writer_res.writerows(res)




def getDataSets():

    stop_list = []
    with open('../data/stop_words.txt', 'r', encoding='utf8') as fr:
        for line in fr:
            if line.strip() != ' ':
                stop_list.append(line.strip())

    jieba.enable_parallel(16)
    with open('../data/medical.csv', 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        for i in reader:
            jieba.add_word(i[0])
    tag_index = {}
    with open('../data/tags.txt','r',encoding='utf8') as fr:
        lines = fr.readlines()
        res_count = 0
        for line in lines:
            tag_index[line.strip()]=res_count
            res_count +=1
    print('标签索引',tag_index)
    data_set = []
    labels = []
    doc = []
    seg = ['。', '？', '！', '?', '!',';','；']
    vocab = {}
    vocab["#PADDING#"] = 0
    vocab["#UNK#"] = 1
    i = 2

    # with open("../data/word_freq.pickle", "rb") as f:
    #     word_freq = pickle.load(f)
    #     for word, freq in word_freq.items():
    #         if freq >= 3 and word not in stop_list:   # 词表去停用词 , 停用词去除 “ 。！？ ”
    #             vocab[word] = i
    #             i += 1
    with open("../data/words.txt", "r",encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
                w = line.strip()
                vocab[w] = i
                i += 1
        # 混合预训练词向量
        print('词表大小：',len(vocab))
        word_vec = np.zeros(shape=(len(vocab)-2,300),dtype=np.float32)

        trained_embedding = load_embedding()   #{词：[]}
        for word in vocab.keys():
            if word != '#PADDING#' and word != '#UNK#':
               if word not in trained_embedding.keys() :
                    word_vec[vocab[word]-2] = np.random.uniform(-0.25,0.25,300)
               else:
                    word_vec[vocab[word]-2] = trained_embedding[word]

    max_sentence_size = 0
    max_word_size = 0
    avg_sentence_size = []
    avg_word_size = []
    with open("../data/PICO_train.txt", 'r', encoding='utf8') as fr:
        reader = fr.readlines()
        docs = []
        labels = []
        doc = []
        label = []
        for line in reader:
            if '###' in line and len(doc)>0 and len(label)>0:
                docs.append(doc)
                labels.append(label)
                avg_sentence_size.append(len(doc))
                if len(doc) > max_sentence_size:
                    max_sentence_size = len(doc)
                doc = []
                label = []
            if '|' in line:
                s = line.strip().split('|')
                sentence = s[2].strip().split()
                label.append(tag_index[s[1]])   # 转换标签对应索引
                if len(sentence) > max_word_size:
                    max_word_size = len(sentence)
                doc.append(sentence)
                avg_word_size.append(len(sentence))

    print("最大句子数", max_sentence_size)
    print("最大词数", max_word_size)
    print('平均句子数', sum(avg_sentence_size)/len(avg_sentence_size))
    print('平均词数', sum(avg_word_size)/len(avg_word_size))

    with open("../data/PICO_dev.txt", 'r', encoding='utf8') as fr:
        reader = fr.readlines()
        docs_dev = []
        labels_dev = []
        doc = []
        label = []
        for line in reader:
            if '###' in line and len(doc)>0 and len(label)>0:
                docs_dev.append(doc)
                labels_dev.append(label)
                doc = []
                label = []
            if '|' in line:
                s = line.strip().split('|')
                sentence = s[2].strip().split()
                label.append(tag_index[s[1]])   # 转换标签对应索引
                doc.append(sentence)

    with open("../data/PICO_test.txt", 'r', encoding='utf8') as fr:
        reader = fr.readlines()
        docs_test = []
        labels_test = []
        doc = []
        label = []
        for line in reader:
            if '###' in line and len(doc)>0 and len(label)>0:
                docs_test.append(doc)
                labels_test.append(label)
                doc = []
                label = []
            if '|' in line:
                s = line.strip().split('|')
                sentence = s[2].strip().split()
                label.append(tag_index[s[1]])   # 转换标签对应索引
                doc.append(sentence)

    max_sentence_size = int(2*sum(avg_sentence_size)/len(avg_sentence_size))
    max_word_size = int(2*sum(avg_word_size)/len(avg_word_size))\

    train = []
    label_train = []
    train_length = []
    data_new = []
    labels_new = []
    for doc in docs:  # train
        # print(doc)
        document = np.zeros((max_sentence_size, max_word_size))
        for i,sent in enumerate(doc):
            if i < max_sentence_size:
                for j,word in enumerate(sent):
                    if j < max_word_size:
                        if word in vocab.keys():
                            document[i][j] = vocab.get(word)
                        else:
                            document[i][j] = vocab.get("#UNK#")
        data_new.append(document)
    for lab in labels:
        if len(lab) < max_sentence_size:
            res = lab + [0]*(max_sentence_size-len(lab))
        else:
            res = lab[:max_sentence_size]
        labels_new.append(res)
    for i in range(len(data_new)):

            train.append(data_new[i])
            label_train.append(labels_new[i])
            train_length.append(len(labels[i]))
    print(len(train))
    print(len(label_train))
    print(len(train_length))

    dev = []
    label_dev = []
    dev_length = []
    data_new = []
    labels_new = []
    for doc in docs_dev:   # dev
        # print(doc)
        document = np.zeros((max_sentence_size, max_word_size))
        for i, sent in enumerate(doc):
            if i < max_sentence_size:
                for j, word in enumerate(sent):
                    if j < max_word_size:
                        if word in vocab.keys():
                            document[i][j] = vocab.get(word)
                        else:
                            document[i][j] = vocab.get("#UNK#")
        data_new.append(document)
    for lab in labels_dev:
        if len(lab) < max_sentence_size:
            res = lab + [0] * (max_sentence_size - len(lab))
        else:
            res = lab[:max_sentence_size]
        labels_new.append(res)
    for i in range(len(data_new)):

        dev.append(data_new[i])
        label_dev.append(labels_new[i])
        dev_length.append(len(labels_dev[i]))

    print(len(dev))
    print(len(label_dev))
    print(len(dev_length))

    test = []
    label_test = []
    test_length = []
    data_new = []
    labels_new = []
    for doc in docs_test:   # test
        # print(doc)
        document = np.zeros((max_sentence_size, max_word_size))
        for i, sent in enumerate(doc):
            if i < max_sentence_size:
                for j, word in enumerate(sent):
                    if j < max_word_size:
                        if word in vocab.keys():
                            document[i][j] = vocab.get(word)
                        else:
                            document[i][j] = vocab.get("#UNK#")
        data_new.append(document)
    for lab in labels_test:
        if len(lab) < max_sentence_size:
            res = lab + [0] * (max_sentence_size - len(lab))
        else:
            res = lab[:max_sentence_size]
        labels_new.append(res)
    for i in range(len(data_new)):
            test.append(data_new[i])
            label_test.append(labels_new[i])
            test_length.append(len(labels_test[i]))
    print(len(test))
    print(len(label_test))
    print(len(test_length))

    with open("../data/test.pickle","wb") as f3:
        pickle.dump(test,f3)
    with open("../data/label_test.pickle","wb") as f4:
        pickle.dump(label_test,f4)
    with open("../data/test_length.pickle","wb") as f5:
        pickle.dump(test_length,f5)

    return train,dev,label_train,label_dev,train_length,dev_length,max_sentence_size,max_word_size, word_vec


def train_embedding():

    stop_list =[]
    with open('./data/stop_words.txt','r',encoding='utf8') as fr:
        for line in fr:
            if line.strip()!=' ':
                stop_list.append(line.strip())

    print(len(stop_list))

    jieba.enable_parallel(16)
    with open('./data/medical.csv', 'r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            # print(i[0])
            jieba.add_word(i[0])
    sentences = []
    with open('./data/corpus.txt','r',encoding='utf8') as fr:
        lines = fr.readlines()
        for line in lines:
            sentence = jieba.lcut(line.strip())
            res = []
            for i in sentence:
                if i not in stop_list:
                    res.append(i)
            if len(res)>0:
                sentences.append(res)
    model = gensim.models.Word2Vec(sentences,size=300,window=5,min_count=0,workers=16)
    # print(model.wv.word_vec('口腔'))

    model.wv.save_word2vec_format('wv300.bin')


def load_embedding():

    data = {}
    with open('../wv300.bin','r') as fr:
        for line in fr:
            aa = line.split(' ')
            # print(len(aa))
            val = aa[1:]
            data[aa[0]] = val
    return data


if __name__=="__main__":

    # train_embedding()
    # load_embedding()

    pre_process()
    # getDataSets()
    # h=0







