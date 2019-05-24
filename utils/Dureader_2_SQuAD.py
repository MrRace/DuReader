# -*- coding: utf-8 -*-
# @CreatTime    : 2019/3/26 11:48
# @Author  : JasonLiu
# @FileName: test.py
# from __future__ import print_function
import json


# SQuAD1.1的数据格式是，一个文档一个json。在文档内部划分为多个paragraphs。每个paragraph对应多个问题和答案。
# 一行行处理，这里的linedata是DuReader的一行数据
# 读取DuReader数据集
# du_reader_file = "E:/share_pro_GPU/MRC2018/data/demo/trainset/search.train.json"


def conver_dureader2squad():
    du_reader_file = "E:/WorkSources/NLP/竞赛/lic2019.ccf.org.cn_机器阅读理解/train_preprocessed/train_preprocessed/trainset/search.train.json"
    train_data = {}
    train_data["data"] = []
    total_count = 0
    opinion_count = 0
    empty_answer_spans_count = 0
    descrip_count = 0
    entity_count = 0
    yes_no_count = 0
    f_result = open("result_dict.txt", "w", encoding="utf8")
    with open(du_reader_file, "r", encoding="utf-8") as f_reader:
        for lidx, line in enumerate(f_reader):
            # 每次加载一个json文件-sample
            total_count = total_count + 1
            # line = line.strip()
            # line = line.encode('utf-8').decode('unicode_escape')
            # line = bytes(line, encoding='utf-8')
            # line = line.decode("utf-8")
            sample = json.loads(line.strip())
            # print(sample)
            question = sample["question"]
            question_type = sample["question_type"]
            # question = question.encode('utf-8').decode('unicode_escape')
            fact_or_opinion = sample["fact_or_opinion"]
            if fact_or_opinion == "OPINION":
                opinion_count = opinion_count + 1
                # continue
            if question_type == "DESCRIPTION":
                descrip_count = descrip_count + 1
            elif question_type == "ENTITY":
                entity_count = entity_count + 1
            elif question_type == "YES_NO":
                yes_no_count = yes_no_count + 1
                # 还需要进一步
                yesno_answers = sample["yesno_answers"]
            else:
                print("ERROR question type=", question_type)
            question_id = sample["question_id"]
            answer_spans = sample["answer_spans"]#这个位置是根据词
            if len(answer_spans) < 1:
                empty_answer_spans_count = empty_answer_spans_count + 1
                # print("answer_spans is empty,total num=", total_count)
                # print("empty_answer_spans_count,=", line.strip())
                continue

            fake_answers = sample["fake_answers"]
            answer_docs = sample["answer_docs"]#answer_spans所对应的document
            if len(answer_docs) > 1:
                print("answer_docs len=", len(answer_docs))
            elif len(answer_docs) == 0:
                print("answer_docs is Empty")
                print(line)
            # DuReader是一个问题多个文档；而SQuAD是在一个文档下提问多个问题，每个段落提出一个问题。
            documents = sample["documents"]
            paragraphs_dic = {}
            paragraphs_dic["paragraphs"] = []
            for d_i in range(len(documents)):
                if d_i != answer_docs[0]:
                    continue
                # 从每个文档中提取出段落。由于这里的文档很短，所以，直接将文档的paragraph连接后作为SQuAD中的段落。
                document = documents[d_i]
                is_selected = document["is_selected"]
                title = document["title"]
                segmented_title = document["segmented_title"]
                paragraphs = document["paragraphs"]# 这是一个list需要将其join起来。
                segmented_paragraphs = document["segmented_paragraphs"]# 每个paragraph的分词结果
                most_related_para = document["most_related_para"]

                context = "".join(paragraphs)
                #
                qas = []
                qas_dic = {}
                qas_dic["answers"] = []
                qas_dic["id"] = str(question_id)
                qas_dic["question"] = question#.decode('utf-8')

                answer = {}
                answer["answer_start"] = answer_spans[0][0]
                if len(fake_answers) > 1:
                    print("fake_answers length=%d,line_num=%d" % (len(fake_answers), total_count))
                answer["text"] = fake_answers[0] # 定位到原文，查找答案
                qas_dic["answers"].append(answer)
                qas.append(qas_dic)

                paragraph = {}
                paragraph["context"] = context
                paragraph["qas"] = qas
                paragraphs_dic["paragraphs"].append(paragraph)

            paragraphs_dic["title"] = question
            train_data["data"].append(paragraphs_dic)
            # if total_count >= 1:
            #     break
        train_data["version"] = 1.1
        # print(json.dumps(train_data, ensure_ascii=False))
        # tt = json.dumps(train_data)
        # print(tt.encode('latin-1').decode('unicode_escape'))
        # tt.encode('utf-8').decode('unicode_escape')
        print("final size=", len(train_data["data"]))
        f_result.write(json.dumps(train_data, ensure_ascii=False))# json.dumps结果是Unicode
        # ensure_ascii=False
        # f_result.write(tt)
    f_result.close()
    print("total_count=", total_count)
    print("empty_answer_spans_count=", empty_answer_spans_count)
    print("opinion_count=", opinion_count)
    # final size = 48099
    # total_count = 136208
    # empty_answer_spans_count = 2260
    # opinion_count = 85849

    """如果将
    final size= 128732
    total_count= 136208
    empty_answer_spans_count= 7476
    opinion_count= 85849
    是否要进一步区分答案类型？？因为是非类型答案还需要考虑字段`yesno_answers`。
    """
def check_convert_result():
    with open("result_dict.txt", "r", encoding="utf8") as conver_file:
        input_data = json.load(conver_file)["data"]
        print(len(input_data))
        for id in range(len(input_data)):
            tt = json.dumps(input_data[id], ensure_ascii=False)
            print(tt)

# conver_dureader2squad()
check_convert_result()
