# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
import pdb
from utils.pretrain_embedding import pre_train
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain word embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--dump_preprocessed_data', action='store_true',
                        help='dump preprocessed data,delete useless fields')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/DuReader2.0/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/DuReader2.0/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/DuReader2.0/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--prepared_dir', default='../data/prepared',
                               help='the dir to store prepared ')
    path_settings.add_argument('--segmented_dir', default='../data/segmented',
                               help='the dir to store segmented sentences')
    path_settings.add_argument('--pretrained_word_path', default=None,
                               help='the pretrained word_embedding file ')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()



def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    whole_filelist = []
    whole_filelist.extend(args.train_files)
    whole_filelist.extend(args.dev_files)
    whole_filelist.extend(args.test_files)
    for data_path in whole_filelist:
        print("data_path=", data_path)
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.prepared_dir, args.segmented_dir, args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files, args.prepared_dir, prepare=True)
    # 此时占用内存27GB
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    if args.pretrain:
        # 基于已有的分词结果训练词向量
        if args.pretrained_word_path:
            # 如果指定了预训练路径，则不需要重新训练，否则根据给定的语料重新训练
            vocab.load_pretrained_embeddings(args.pretrained_word_path)
        else:
            # 训练耗时20分钟左右
            pre_train(brc_data, args.segmented_dir)
            vocab.load_pretrained_embeddings(os.path.join(args.segmented_dir, 'w2v_dic.data'))
    else:
        vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
    del vocab
    pdb.set_trace()
    if args.dump_preprocessed_data:
        logger.info('Dump preprocessed datasets...')
        # 注意，如果BRCDataset初始化时候没有清空无关字段，则dump结果过大
        with open(os.path.join(args.prepared_dir, 'train_set.pkl'), 'wb') as f_train_out:
            pickle.dump(brc_data.train_set, f_train_out)
        del brc_data.train_set
        logger.info('Dump train_set.pkl...')

        with open(os.path.join(args.prepared_dir, 'dev_set.pkl'), 'wb') as f_dev_out:
            pickle.dump(brc_data.dev_set, f_dev_out)
        del brc_data.dev_set
        logger.info('Dump dev_set.pkl...')

        with open(os.path.join(args.prepared_dir, 'test_set.pkl'), 'wb') as f_test_out:
            pickle.dump(brc_data.test_set, f_test_out)
        del brc_data.test_set
        logger.info('Dump test_set.pkl...')


    logger.info('Done with preparing!')

# 如果采用外部词向量是否加快？？

def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, prepared_dir=args.prepared_dir)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files, prepared_dir=args.prepared_dir)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files, prepared_dir=args.prepared_dir)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
