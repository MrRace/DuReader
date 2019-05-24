# -*- coding: utf-8 -*-
# @CreatTime    : 2019/5/9 10:59
# @Author  : JasonLiu
# @FileName: test_write_load_tfrecord.py
"""
测试tfrecor的生成和读取
"""
import tensorflow as tf
import json
import os
import numpy as np
import pdb
import sys
sys.path.append('..')
from matplotlib import pyplot as plt
from utils.pretrain_embedding import pre_train
from collections import Counter
# 第一步：生成TFRecord Writer

train = True
tfrecord_path = "../data/test_write_read_tf/"
txt_filename = "../data/DuReader2.0/trainset/search.train.json"
max_p_len = 500

"""
需要先遍历train、test和dev 数据集构建vocab

需要保持的字段是answers，但是为了整个数据以tfrecord格式保存，需要将整个str转为int（查字典的方式）
"""

import pickle
import gc
import logging
from dataset import BRCDataset
from vocab import Vocab

logger = logging.getLogger("QANet")

def gen_vocab():
    """
    一次性遍历数据集，生成vocab。该vocab是为了后期将token转为id
    :return:
    """
    max_p_num = 5
    max_p_len = 500
    max_q_len = 60
    embed_size = 300
    pretrain = True
    pretrained_word_path = ""
    train_files = ['../data/DuReader2.0/trainset/search.train.json', '../data/DuReader2.0/trainset/zhidao.train.json']
    dev_files = ['../data/DuReader2.0/devset/search.dev.json', '../data/DuReader2.0/devset/zhidao.dev.json']
    test_files = ['../data/DuReader2.0/testset/search.test.json', '../data/DuReader2.0/testset/zhidao.test.json']
    prepared_dir = "../data/test_write_read_tf/prepared"
    vocab_dir = "../data/test_write_read_tf/vocab"
    segmented_dir = "../data/test_write_read_tf/segmented"

    for dir_path in [prepared_dir, segmented_dir, vocab_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # pdb.set_trace()
    brc_data = BRCDataset(max_p_num, max_p_len, max_q_len, train_files, dev_files, test_files, prepared_dir, prepare=True)

    # 此时占用内存27GB
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))

    print('Assigning embeddings...')
    if pretrain:
        # 基于已有的分词结果训练词向量
        if pretrained_word_path:
            # 如果指定了预训练路径，则不需要重新训练，否则根据给定的语料重新训练
            vocab.load_pretrained_embeddings(pretrained_word_path)
        else:
            # 训练耗时20分钟左右
            # pre_train(brc_data, segmented_dir) 预训练和生成vocab是分开的。严格上来说，预训练包括了vocab的生成
            vocab.load_pretrained_embeddings(os.path.join(segmented_dir, 'w2v_dic.data'))
    else:
        vocab.randomly_init_embeddings(embed_size)

    print('Saving vocab,only train data...')
    with open(os.path.join(vocab_dir, 'vocab_traindata_pretrainW2V.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    del brc_data.train_set

    # 加入test data
    print('Saving vocab,train + test data...')
    for word in brc_data.word_iter('test'):
        vocab.add(word)
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    print('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    with open(os.path.join(vocab_dir, 'vocab_train_test_data.data'), 'wb') as fout1:
        pickle.dump(vocab, fout1)

    # 加入dev data
    print('Saving vocab,train + test + dev data...')
    for word in brc_data.word_iter('dev'):
        vocab.add(word)
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    print('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))
    with open(os.path.join(vocab_dir, 'vocab_train_test_dev_data.data'), 'wb') as fout2:
        pickle.dump(vocab, fout2)

    logger.info('Done with preparing!')

def gen_pretrain():
    """
    基于语料训练词向量，采用Word2Vec。
    :return:
    """
    max_p_num = 5
    max_p_len = 500
    max_q_len = 60
    embed_size = 300

    train_files = ['../data/DuReader2.0/trainset/search.train.json', '../data/DuReader2.0/trainset/zhidao.train.json']
    dev_files = ['../data/DuReader2.0/devset/search.dev.json', '../data/DuReader2.0/devset/zhidao.dev.json']
    test_files = ['../data/DuReader2.0/testset/search.test.json', '../data/DuReader2.0/testset/zhidao.test.json']
    segmented_dir = "../data/DuReader2.0/whole_segmented"

    for dir_path in [segmented_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 处理训练集
    dataloader = BRCDataset(max_p_num, max_p_len, max_q_len, train_files, dev_files=[], test_files=[],
                            prepared_dir="", prepare=True)
    dataloader.save_segments(seg_dir=segmented_dir, train_files=train_files)
    del dataloader

    # 处理 dev set
    dataloader = BRCDataset(max_p_num, max_p_len, max_q_len, train_files=[], dev_files=dev_files,
                            test_files=[], prepared_dir="", prepare=True)
    dataloader.save_segments(seg_dir=segmented_dir, dev_files=dev_files)
    del dataloader

    # 处理 test set
    dataloader = BRCDataset(max_p_num, max_p_len, max_q_len, train_files=[], dev_files=[],
                            test_files=test_files, prepared_dir="", prepare=True)
    dataloader.save_segments(seg_dir=segmented_dir, test_files=test_files)
    del dataloader

    pre_train(segmented_dir, embed_size)

    # 预训练结束后，直接生成vocab.data文件(包含有vocab和词向量)


def gen_vocab_pretrain_w2v():
    """
    在预训练的过程也将结果写到vocab中
    :return:
    """
    max_p_num = 5
    max_p_len = 500
    max_q_len = 60
    embed_size = 300
    pretrain = True
    pretrained_word_path = "" # 是否采用第三方预训练的词向量
    train_files = ['../data/DuReader2.0/trainset/search.train.json', '../data/DuReader2.0/trainset/zhidao.train.json']
    dev_files = ['../data/DuReader2.0/devset/search.dev.json', '../data/DuReader2.0/devset/zhidao.dev.json']
    test_files = ['../data/DuReader2.0/testset/search.test.json', '../data/DuReader2.0/testset/zhidao.test.json']
    prepared_dir = ""
    vocab_dir = "../data/test_write_read_tf/vocab"
    segmented_dir = "../data/DuReader2.0/whole_segmented"

    for dir_path in [segmented_dir, vocab_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    brc_data = BRCDataset(max_p_num, max_p_len, max_q_len, train_files, dev_files, test_files, prepared_dir, prepare=True)

    # 此时占用内存27GB
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))

    print('Assigning embeddings...')
    if pretrain:
        # 基于已有的分词结果训练词向量
        if pretrained_word_path:
            # 如果指定了预训练路径，则不需要重新训练，否则根据给定的语料重新训练
            vocab.load_pretrained_embeddings(pretrained_word_path)
        else:
            vocab.load_pretrained_embeddings(os.path.join(segmented_dir, 'w2v_dic.data'))
    else:
        vocab.randomly_init_embeddings(embed_size)

    print('Saving vocab,only train data...')
    with open(os.path.join(vocab_dir, 'vocab_traindata_pretrainW2V.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

"""
预处理方式，将训练集转为tfrecord格式，训练时候直接读取tfrecord格式
"""

def save_tfrecord_sequence_example():
    """
    在预处理期间也完成token_id的转换
    :return:
    """
    line_count = 0
    set_line_num = 100000
    vocab_dir = "../data/test_write_read_tf/vocab"
    with open(os.path.join(vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    tfrecord_filename = os.path.join(tfrecord_path, "train.tfrecord")
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    with open(txt_filename, encoding='utf8') as fin:
        data_set = []
        for lidx, line in enumerate(fin):
            sample = json.loads(line.strip())
            line_count = line_count + 1
            del sample['question']

            ex = tf.train.SequenceExample()
            str_list = []
            # feature = {}
            answers_feature = []
            # for i in range(len(sample["answers"])):
            #     str_list.append(sample["answers"][i].encode('utf-8'))
            #     ex.context.feature["answers"].bytes_list.value.append(sample["answers"][i].encode('utf-8'))
            #     answer_feature =

            # feature = {"answers": tf.train.Feature(bytes_list=tf.train.BytesList(value=str_list))}
            feature = {"answers": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans.encode('utf-8') for ans in sample["answers"]]))}

            ex_ques_token_ids = ex.feature_lists.feature_list["passage_token_ids"]#list中还有list
            ex_pass_token_ids = ex.feature_lists.feature_list["passage_token_ids"]#list中还有list

            if train:
                # testdata 是没有这个字段的
                if len(sample['answer_spans']) == 0:
                    continue
                if sample['answer_spans'][0][1] >= max_p_len:
                    continue
                feature["answer_spans"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["answer_spans"][0]))
                ex.context.feature["answer_spans"].int64_list.value.extend(sample["answer_spans"][0])

            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']
                feature['answer_passages'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample['answer_docs']))
                del sample['answer_docs']
                del sample['fake_answers']
                del sample['segmented_answers']
            else:
                print("not answer docs")
            question_tokens = sample['segmented_question']
            # print("question_tokens=", question_tokens)
            ques_token_ids = vocab.convert_to_ids(question_tokens)
            # print("ques_token_ids=", ques_token_ids)
            recover_from_ids = vocab.recover_from_ids(ques_token_ids)
            # print("recover_from_ids,words=", recover_from_ids)

            feature["question_token_ids"] = tf.train.Feature(int64_list=
                                                             tf.train.Int64List(value=ques_token_ids))
            ex.context.feature["question_token_ids"].int64_list.value.extend(ques_token_ids)
            customer = tf.train.Features(feature=feature)

            sample['passages'] = []
            pass_list = []
            pass_token_list = []
            is_selected_list = []
            for d_idx, doc in enumerate(sample['documents']):
                del doc['title']
                del doc['segmented_title']
                del doc['paragraphs']
                if train:
                    # 预处理阶段的训练数据
                    most_related_para = doc['most_related_para']
                    passage_tokens = doc['segmented_paragraphs'][most_related_para]  # word_tokenize
                    pass_token_ids = vocab.convert_to_ids(passage_tokens)
                    pass_token_ids_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=pass_token_ids))
                    # pass_token_ids_fealist = tf.train.FeatureList(feature=[pass_token_ids_feature])
                    pass_token_list.append(pass_token_ids_feature) # token_id_feature是单个feature
                    is_selected_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[doc['is_selected']]))
                    is_selected_list.append(is_selected_feature)
                    # 由于一个问题，参考了多个document，所以`passages`字段是一个数组
                    sample['passages'].append(
                        {'passage_token_ids': pass_token_ids,
                         'is_selected': doc['is_selected']}
                    )
                    pass_feature = {
                                    'passage_token_ids': pass_token_ids_feature,#一个feature
                                    'is_selected': is_selected_feature#是一个feature
                                    }
                    # 如此pass_feature是一个features
                    pass_list.append(pass_feature)
                else:
                    pass
            # feature["passages"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["passages"]))#不支持字典
            # feature["passages"] = tf.train.FeatureList(feature=pass_list)#会报错，大概是因为一个feature不能存featurelist,
            # pass_feature_list = tf.train.FeatureList(feature=pass_list)  # 大概是因为一个feature不能存featurelist
            pass_token_list_feature = tf.train.FeatureList(feature=pass_token_list)
            is_selected_list_feature = tf.train.FeatureList(feature=is_selected_list)
            ids = tf.train.FeatureLists(feature_list={
                                        'passage_token_ids': pass_token_list_feature,
                                        'is_selected': is_selected_list_feature
                                        })
            #
            example = tf.train.SequenceExample(context=customer, feature_lists=ids)
            # features = tf.train.Features(feature=feature)
            # example = tf.train.Example(features=features)
            # print(example)
            # pdb.set_trace()
            # writer.write(example.SerializeToString())
            # feature["passages"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["passages"]))
            # 第四步：可以理解为将内层多个feature的字典数据再编码，集成为features
            # features = tf.train.Features(feature=feature)
            # features = tf.train.FeatureList(feature=feature)
            # example = tf.train.Example(features=features)
            # 第六步：将example数据序列化为字符串
            Serialized = example.SerializeToString()
            # 第七步：将序列化的字符串数据写入协议缓冲区
            # print("Serialized=", Serialized)#应该是byte类型
            # pdb.set_trace()
            # example_proto = tf.train.Example.FromString(Serialized)
            # print("example_proto", example_proto)
            writer.write(Serialized)
            del feature
            del sample['documents']
            del sample['segmented_question']
            data_set.append(sample)
    # 记得关闭writer和open file的操作
    writer.close()

def save_tfrecord_example():
    """
    将所有的字段都写到feature中，不像此前那样：采用2个字段context和feature_lists
    :return:
    """
    line_count = 0
    set_line_num = 100000
    vocab_dir = "../data/test_write_read_tf/vocab"
    with open(os.path.join(vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    tfrecord_filename = os.path.join(tfrecord_path, "train_example.tfrecord")
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    with open(txt_filename, encoding='utf8') as fin:
        data_set = []
        for lidx, line in enumerate(fin):
            sample = json.loads(line.strip())
            line_count = line_count + 1
            del sample['question']

            ex = tf.train.SequenceExample()
            feature = {"answers": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans.encode('utf-8') for ans in sample["answers"]]))}

            if train:
                # testdata 是没有这个字段的
                if len(sample['answer_spans']) == 0:
                    continue
                if sample['answer_spans'][0][1] >= max_p_len:
                    continue
                feature["answer_spans"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["answer_spans"][0]))
                ex.context.feature["answer_spans"].int64_list.value.extend(sample["answer_spans"][0])

            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']
                feature['answer_passages'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample['answer_docs']))
                del sample['answer_docs']
                del sample['fake_answers']
                del sample['segmented_answers']
            else:
                print("not answer docs")
            question_tokens = sample['segmented_question']
            ques_token_ids = vocab.convert_to_ids(question_tokens)
            feature["question_token_ids"] = tf.train.Feature(int64_list=
                                                             tf.train.Int64List(value=ques_token_ids))
            ex.context.feature["question_token_ids"].int64_list.value.extend(ques_token_ids)
            # customer = tf.train.Features(feature=feature)

            sample['passages'] = []
            pass_list = []
            pass_token_list = []
            is_selected_list = []
            for d_idx, doc in enumerate(sample['documents']):
                del doc['title']
                del doc['segmented_title']
                del doc['paragraphs']
                if train:
                    # 预处理阶段的训练数据
                    most_related_para = doc['most_related_para']
                    passage_tokens = doc['segmented_paragraphs'][most_related_para]  # word_tokenize
                    pass_token_ids = vocab.convert_to_ids(passage_tokens)
                    pass_token_ids_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=pass_token_ids))
                    # pass_token_ids_fealist = tf.train.FeatureList(feature=[pass_token_ids_feature])
                    pass_token_list.append(pass_token_ids_feature) # token_id_feature是单个feature
                    is_selected_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[doc['is_selected']]))
                    is_selected_list.append(is_selected_feature)
                    # 由于一个问题，参考了多个document，所以`passages`字段是一个数组
                    sample['passages'].append(
                        {'passage_token_ids': pass_token_ids,
                         'is_selected': doc['is_selected']}
                    )
                    pass_feature = {
                                    'passage_token_ids': pass_token_ids_feature,#一个feature
                                    'is_selected': is_selected_feature#是一个feature
                                    }
                    # 如此pass_feature是一个features
                    pass_list.append(pass_feature)
                else:
                    pass
            # feature["passages"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["passages"]))#不支持字典
            # feature["passages"] = tf.train.FeatureList(feature=pass_list)#会报错，大概是因为一个feature不能存featurelist,
            # pass_feature_list = tf.train.FeatureList(feature=pass_list)  # 大概是因为一个feature不能存featurelist
            pass_token_list_feature = tf.train.FeatureList(feature=pass_token_list)
            is_selected_list_feature = tf.train.FeatureList(feature=is_selected_list)
            ids = tf.train.FeatureLists(feature_list={
                                        'passage_token_ids': pass_token_list_feature,
                                        'is_selected': is_selected_list_feature
                                        })
            #
            example = tf.train.SequenceExample(context=customer, feature_lists=ids)
            # features = tf.train.Features(feature=feature)
            # example = tf.train.Example(features=features)
            # print(example)
            # pdb.set_trace()
            # writer.write(example.SerializeToString())
            # feature["passages"] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample["passages"]))
            # 第四步：可以理解为将内层多个feature的字典数据再编码，集成为features
            # features = tf.train.Features(feature=feature)
            # features = tf.train.FeatureList(feature=feature)
            # example = tf.train.Example(features=features)
            # 第六步：将example数据序列化为字符串
            Serialized = example.SerializeToString()
            # 第七步：将序列化的字符串数据写入协议缓冲区
            # print("Serialized=", Serialized)#应该是byte类型
            # pdb.set_trace()
            # example_proto = tf.train.Example.FromString(Serialized)
            # print("example_proto", example_proto)
            writer.write(Serialized)
            del feature
            del sample['documents']
            del sample['segmented_question']
            data_set.append(sample)
    # 记得关闭writer和open file的操作
    writer.close()



def extract_fn(data_record):

    con_fea = {
        'answers': tf.VarLenFeature(dtype=tf.string),
        'answer_spans': tf.FixedLenFeature([2], dtype=tf.int64),# 不写2的话，会报错
        'answer_passages': tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fea = {
         'passage_token_ids': tf.VarLenFeature(dtype=tf.int64),
         'is_selected': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    # 把序列化样本和解析字典送入函数里得到解析的样本
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=data_record, context_features=con_fea, sequence_features=seq_fea)
    context_parsed["answers"] = tf.sparse_tensor_to_dense(context_parsed["answers"], default_value="0")
    sequence_parsed["passage_token_ids"] = tf.sparse_tensor_to_dense(sequence_parsed["passage_token_ids"], default_value=0)
    # 其实这里的data[0]就是context_parsed，data[1]就是sequence_parsed

    # context_data, sequence_data
    # print("type context_data=", type(context_data))
    # print("type sequence_data=", type(sequence_data))
    # print('Context:')
    # for name, tensor in context_data.items():
    #     # print('{}: {}'.format(name, tensor.eval(session=sess)))
    #     # print('{}: {}'.format(name, sess.run(tensor)))
    #     print(name)
    #     print(sess.run(tensor))
    # print('\nData')
    # for name, tensor in sequence_data.items():
    #     print('{}: {}'.format(name, tensor.eval(session=sess)))
    rs_dic = {}
    rs_dic['answers'] = context_parsed['answers']
    rs_dic['answer_spans'] = context_parsed['answer_spans']
    rs_dic['answer_passages'] = tf.expand_dims(tf.convert_to_tensor(context_parsed['answer_passages']), 0)
    # 注意，如果不进行维度变换的话，其shape是(),因为是scale值
    rs_dic['passage_token_ids'] = sequence_parsed['passage_token_ids']
    rs_dic['is_selected'] = sequence_parsed['is_selected']
    return rs_dic

def load_tfrecord():
    """
    生成一个batch
    :return:
    """
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        temp = example.ParseFromString(string_record)
        print(example)
        print("feature=", dict(example.features.feature))
        print("feature_str=", example.features.feature['answers'].bytes_list.value)
        answers = example.features.feature['answers'].bytes_list.value
        print(type(answers))#google.protobuf.pyext._message.RepeatedScalarContainer
        t_str = example.features.feature['answers']
        print(type(t_str))
        answers_list = list(answers)
        for i in range(len(answers_list)):
            print(answers_list[i].decode("utf-8"))
        # Exit after 1 iteration as this is purely demonstrative.
        break


def read_and_decode():
    """
    可以打印数据，需要decode(utf-8)，
    :return:
    """
    filename = tfrecord_filename
    filename_queue = tf.train.string_input_producer([filename])#这个使用方法是要被淘汰的

    reader = tf.data.TFRecordDataset()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                                 'answers': tf.VarLenFeature(dtype=tf.string),
                                                  'answer_spans': tf.FixedLenFeature([2], dtype=tf.int64),
                                                })
    print(features)
    # decoded = tf.decode_raw(features['answers'], tf.uint8)
    # print(tf.shape(decoded)[0])

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        print(session.run(features["answer_spans"]))
        print(session.run(features["answers"]))

    # dataset = tf.data.TFRecordDataset([tfrecord_filename])
    # dataset = dataset.map(extract_fn)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)  # 初始化
    #     try:
    #         while True:
    #             data_record = sess.run(next_element)
    #             answers = tf.decode_raw(data_record['answers'], tf.string)
    #             print(answers)
    #             # print("dataset=", sess.run([data_record['answers'],
    #             #       data_record['answer_spans']]))
    #             break
    #     except:
    #         pass

def parse_sequence_example(serialized):
    con_fea = {
        'answers': tf.VarLenFeature(dtype=tf.string),
        'answer_spans': tf.FixedLenFeature([2], dtype=tf.int64),  # 不写2的话，会报错
        'answer_passages': tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fea = {
        'passage_token_ids': tf.VarLenFeature(dtype=tf.int64),
        'is_selected': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    # 是否可以忽略某些暂时用不到的字段？

    context, sequence = tf.parse_single_sequence_example(serialized=serialized,
                                                         context_features=con_fea,
                                                         sequence_features=seq_fea)
    context["answers"] = tf.sparse_tensor_to_dense(context["answers"], default_value="0")
    sequence["passage_token_ids"] = tf.sparse_tensor_to_dense(sequence["passage_token_ids"], default_value=0)
    return context, sequence


def deflate(x):
    '''
    Undo Hack. We undo the expansion we did in expand
    '''

    x['answer_passages'] = tf.squeeze(x['answer_passages'])
    return x

def read_tfrecord():
    """
    已测试通过:能够将tfrecord数据进行读取
    :return:
    """
    sess = tf.InteractiveSession()
    tfrecord_filename = os.path.join(tfrecord_path, "train.tfrecord")
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    new_dataset = dataset.map(extract_fn)
    print("new_dataset=", new_dataset)
    print("new_dataset type=", type(new_dataset))

    filename_queue = tf.train.string_input_producer([tfrecord_filename], shuffle=False, num_epochs=5)
    reader = tf.TFRecordReader()
    _, serialized_ex = reader.read(filename_queue)
    context_parsed, sequence_parsed = parse_sequence_example(serialized_ex)
    print("context_parsed=", context_parsed)
    print("sequence_parsed=", sequence_parsed)

    # return
    # pdb.set_trace()
    #,  为何answer_span字段也要padded，不能够
    #'answer_spans': [2]


    # 创建获取数据集中样本的迭代器
    iterator = new_dataset.make_one_shot_iterator()
    # # 获得下一个样本
    # next_element = iterator.get_next()
    # print("next_element type=", type(next_element))
    # print("next_element 0 type=", type(next_element[0]))
    # print("next_element 0=", next_element[0])
    # print("next_element 1 type=", type(next_element[1]))
    # print("next_element 1=", next_element[1])

    next_element = iterator.get_next()
    print("answer_passages type=", type(next_element['answer_passages']))
    print("answer_passages value=", next_element['answer_passages'])

    shuffle_dataset = new_dataset.shuffle(buffer_size=10000)  # 数据进行打乱
    """
    con_fea = {
        'answers': tf.VarLenFeature(dtype=tf.string),
        'answer_spans': tf.FixedLenFeature([2], dtype=tf.int64),# 不写2的话，会报错
        'answer_passages': tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fea = {
         'passage_token_ids': tf.VarLenFeature(dtype=tf.int64),
         'is_selected': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    """
    # batch 方法1,已测试通过
    # batch_dataset = new_dataset.padded_batch(4, padded_shapes={'answers': [None],
    #                                                            'answer_spans': [2],
    #                                                            'answer_passages': 1,
    #                                                            'passage_token_ids': tf.TensorShape([None, None]),
    #                                                            'is_selected': [None]
    #                                                            })
    # batch_dataset = batch_dataset.map(deflate)

    # # batch 方法2
    batch_dataset = new_dataset.apply(tf.data.experimental.unbatch())#ValueError: Cannot unbatch an input with scalar components.
    batch_dataset = batch_dataset.batch(4)

    # batch_padding_dataset = new_dataset.padded_batch(4, padded_shapes={'answers': [None], 'answer_spans': [None]})

    # 手动构建batch，对所有需要用到的数据手动组成一个数组
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 12 * 4
    # batch_dataset = tf.data.Dataset.batch([context_parsed, sequence_parsed], batch_size=4)

    # batch_dataset = tf.train.batch(tensors=[new_dataset], batch_size=4, dynamic_pad=True)
    # batch_dataset = batch_dataset.batch(4)#数据再进行batch size
    # 不进行padding的话InvalidArgumentError (see above for traceback): Cannot batch tensors with different shapes in component 2.
    # First element had shape [3] and element 1 had shape [1]


    shuffle_iterator = shuffle_dataset.make_one_shot_iterator()
    shuffle_next_element = shuffle_iterator.get_next()

    batch_iterator = batch_dataset.make_one_shot_iterator()
    pdb.set_trace()
    # batch_iterator = batch_dataset.make_initializable_iterator()
    batch_next_element = batch_iterator.get_next()

    i = 1
    while True:
        # 不断的获得下一个样本
        try:
            # 获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂
            answers, answer_spans = sess.run([next_element['answers'], next_element['answer_spans']])#
            # pdb.set_trace()
            shuffle_answers, shuffle_answer_spans = sess.run([shuffle_next_element['answers'], shuffle_next_element['answer_spans']])
            batch_answers, batch_answers_spans, batch_answer_pass, batch_pass_token_ids, is_selected = \
                sess.run([batch_next_element['answers'], batch_next_element['answer_spans'],
                          batch_next_element['answer_passages'], batch_next_element['passage_token_ids'],
                          batch_next_element['is_selected']])
            # batch_answer_pass如果没有再变换回到之前，则为2维的矩阵。做了一次变换之后，则为一维的
        # 如果遍历完了数据集，则返回错误
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
        else:
            # 显示每个样本中的所有feature的信息，只显示scalar的值
            print('==============example %s ==============' % i)
            # 两者的type都是class 'numpy.ndarray'
            print("answer_spans shape=%s | type=%s" % (answer_spans.shape, type(answer_spans)))
            print("answer_spans=", answer_spans)
            print("answers shape=%s | type= %s" % (answers.shape, type(answers)))
            for el in answers:
                print(el.decode('UTF-8'))
            # print("answers=", answers)
            # print(np.array2string(answers))#这种方法貌似不行
            # temp = [x.decode('utf-8') for x in answers]
            print("shuffle_answer_spans=", shuffle_answer_spans)
            temp = [x.decode('utf-8') for x in shuffle_answers]
            print("shuffle_answers=", temp)

            print("batch_answers shape=", batch_answers.shape)
            print("batch_answers_spans shape=", batch_answers_spans.shape)
            print("batch_pass_token_ids shape=", batch_pass_token_ids.shape)
            print("batch_is_selected shape=", is_selected.shape)
            pdb.set_trace()
            # if batch_answers.shape[0] != batch_answers_spans.shape[0]:
            #     print("ERROR")
            #     break
            #
            # for batch_index in range(batch_answers.shape[0]):
            #     bt_example_answers = batch_answers[batch_index]#其中每个元素都是numpy.ndarray
            #     bt_example_spans = batch_answers_spans[batch_index]
            #
            #     templist = list(bt_example_answers)
            #     templ = bt_example_answers.tolist()
            #
            #     print("batch_index=%d, each example' spans=%s" % (batch_index, str(bt_example_spans)))
            #     print("batch_index=%d,each example'answers shape=%s" % (batch_index, str(bt_example_answers.shape)))
            #     for ans_index in range(len(templ)):
            #         print("ans_index=%d,answers=%s" % (ans_index, templ[ans_index].decode('UTF-8')))



                # for span_index in range(batch_answers_spans.shape[1]):
                #     print("batch_index=%d, each example' spans=%s" % (span_index, str(bt_example_spans)))

            # print("answers.astype(str)=", answers.astype(str))

            # print('answers shape=%s | value: %s' % (answers, answers.shape))
            # print('answer_spans shape: %s | type: %s' % (answer_spans.shape, answer_spans.dtype))

            break

        i += 1


# load_tfrecord()
# gen_vocab() # 生成vocab
# save_tfrecord_sequence_example() # 转为tfrecord
# save_tfrecord_example() #另一种方式的write到tfrecord
# gen_pretrain() # 基于语料进行预训练，word2vec。后续尝试Glove等其他预训练方式
gen_vocab_pretrain_w2v()
# read_and_decode()
# read_tfrecord()