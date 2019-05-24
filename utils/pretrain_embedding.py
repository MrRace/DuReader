import sys
import os
import logging
import pathlib
import numpy as np
import pdb
from gensim.models import word2vec
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import gensim

# def save_seg_data(brc_data, tar_dir):
#     """
#     将问句和原文的token以空格为间隔拼接，写到segmented_dir文件夹中，对应结果后缀为.seg
#     :param brc_data:
#     :param tar_dir:
#     :return:
#     """
#     # print('Converting ' + file)
#     # fin = open(file, encoding='utf8')
#     out_file = os.path.join(tar_dir, 'train_set.seg')
#     with open(out_file, 'w', encoding='utf8') as ftrain:
#         for sample in brc_data.train_set:
#             ftrain.write(' '.join(sample['segmented_question']) + '\n')
#             for passage in sample['passages']:
#                 ftrain.write(' '.join(passage['passage_tokens']) + '\n')
#             del sample
#     ftrain.close()
#
#     out_file = os.path.join(tar_dir, 'dev_set.seg')
#     with open(out_file, 'w', encoding='utf8') as fdev:
#         for sample in brc_data.dev_set:
#             fdev.write(' '.join(sample['segmented_question']) + '\n')
#             for passage in sample['passages']:
#                 fdev.write(' '.join(passage['passage_tokens']) + '\n')
#             del sample
#     fdev.close()
#
#     out_file = os.path.join(tar_dir, 'test_set.seg')
#     with open(out_file, 'w', encoding='utf8') as ftest:
#         for sample in brc_data.test_set:
#             ftest.write(' '.join(sample['segmented_question']) + '\n')
#             for passage in sample['passages']:
#                 ftest.write(' '.join(passage['passage_tokens']) + '\n')
#             del sample
#     ftest.close()


def pre_train(segmented_dir, embed_size):
    """
    根据训语料训练词向量。或者可以考虑全部语料加上百度知道的数据集？？
    :param brc_data:
    :param segmented_dir:
    :return:
    """

    sys.path.append('..')
    # 将原始数据的分词结果进行保存
    # save_seg_data(brc_data, segmented_dir)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # 这里的语料是被预处理成多个结果PathLineSentences可以支持多个大文件，对内存很友好。
    # 如果语料是单个大文件的话，建议使用LineSentences（file）这个类加载训练语料，同样内存友好。
    # 默认embed_size=300
    model = word2vec.Word2Vec(word2vec.PathLineSentences(segmented_dir), size=embed_size, min_count=2, workers=12, iter=10)
    # 保存模型
    model.save(os.path.join(segmented_dir, 'w2v_dic.data'))
    with open(os.path.join(segmented_dir, 'w2v_dic.data'), 'w', encoding='utf-8') as f:
        for word in model.wv.vocab:
            f.write(word + ' ')
            f.write(' '.join(list(map(str, model[word]))))
            f.write('\n')
    f.close()

# 衡量预训练的效果
def check_word2vec():
    """

    :return:
    """
    #
    model = gensim.models.KeyedVectors.load_word2vec_format('./w2v_dic.data', binary=False, unicode_errors='ignore')
    val = model.wv.vocab
    index2word = model.index2word
    vectors = model.vectors
    print(index2word[2000])
    print(model.most_similar('喷子'))
    # model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

#gensim.scripts.word2vec2tensor
def gen_metadata_for_tensorboard():
    """
    为了使用tensorboard，需要将已经训练好的词向量做处理.
    主要做的是将词嵌入以tf变量保存

    运行完程序后：
    tensorboard --logdir=metadata_dir

    已测试通过，正常使用
    :return: 
    """
    from tqdm import tqdm
    word2vec_path = ""
    word2vec_name = ""
    save_metadata_dir = ""
    word2vec_file = os.path.join(word2vec_path, word2vec_name)
    with open(word2vec_file, 'r') as f:
        header = f.readline()#词典大小 词向量维度
        vocab_size, vector_size = map(int, header.split())
        words, embeddings = [], []
        for _ in tqdm(range(vocab_size)):
            word_list = f.readline().split(' ')
            word = word_list[0]
            vector = word_list[1:]
            words.append(word)
            embeddings.append(np.array(vector))

    # 将词向量转为tensorboard需要的格式
    with tf.Session() as sess:
        # tf.assign():这里是一个将具体数值（即，词向量矩阵）赋值给tf Variable的例子：
        embed_martix = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[len(words), vector_size])
        set_x = tf.assign(embed_martix, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        # # 需要保存一个metadata文件,给词典里每一个词分配一个身份
        with open(os.path.join(save_metadata_dir, "metadata.tsv"), 'w') as f:
            for word in tqdm(words):
                f.write(word + '\n')

        saver = tf.train.Saver()
        save_path = saver.save(sess, "model_dir/model.ckpt")
        print("Model saved in path: %s" % save_path)
        # 写 TensorFlow summary
        summary_writer = tf.summary.FileWriter(save_metadata_dir, sess.graph)
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = 'embedding'
        embedding_conf.metadata_path = 'metadata.tsv'#注意这里的路径，否则可能出现如下错误
        #"/data/liujiepeng/NLP/MachineComprehension/DuReader/data/DuReader2.0/segmented/metadata_dir/./metadata_dir/metadata.tsv" not found, or is not a file
        projector.visualize_embeddings(summary_writer, config)

        # 保存模型
        # word2vec参数的单词和词向量部分分别保存到了metadata和ckpt文件里面
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_metadata_dir, "model.ckpt"))



def gen_tensorboard_from_w2vdata():
    """
    将已有的词向量转为tensorboard可渲染的数据。
    主要是生成模型文件(包含各个点信息)和metadata文件(包含各个点的label信息)。
    已测试，正常使用
    :return:
    """
    model_path = ""
    model_name = ""
    model = gensim.models.KeyedVectors.load_word2vec_format('./w2v_dic.data', binary=False, unicode_errors='ignore')
    max_size = len(model.wv.vocab)#为何减去1？？
    vocab_size = len(model.vocab)
    print("max_size=", max_size)
    print("vocab_size=", vocab_size)
    print("model.vector_size=", model.vector_size)
    w2v = np.zeros((max_size, model.vector_size))
    # 保存词典
    path = "tensorboard"
    with open(os.path.join(path, "metadata.tsv"), "w+", encoding='utf-8') as file_metadata:
        for i, word in enumerate(model.wv.index2word[:max_size]):
            w2v[i] = model.wv[word]
            file_metadata.write(word + "\n")

    sess = tf.InteractiveSession()
    with tf.device("/cpu:0"):
        embedding = tf.Variable(w2v, trainable=False, name="embedding")#存储embedding
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path, sess.graph)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = "embedding"#注意这里的名字要与上述的变量一致
        embed.metadata_path = "metadata.tsv"
        projector.visualize_embeddings(writer, config)
        saver.save(sess, path + "/model.ckpt", global_step=max_size)


def word2vec_visualize():
    """

    测试通过，可以正常使用
    :return:
    """
    model_path = "./w2v_dic.data"
    output_path = "./metadata_dir"
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    meta_file = "metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

if __name__ == '__main__':
    check_word2vec()