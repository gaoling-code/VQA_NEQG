import argparse
import json
from collections import Counter
import itertools

import config
import data
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
        将tokens列表的迭代对象转换为词汇表
        这些tokens可以是单个答案，也可以是问题中的word tokens。
    """
    # itertools.chain.from_iterable:将多个迭代器连接成一个统一的迭代器的最高效的方法
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)       # Counter:计数功能
    if top_k:                   # 针对answer的处理，取 top-k commen的
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:                       # 针对question的处理
        most_common = counter.keys()
    # descending in count, then lexicographical order
    # 先按计数递减，再按字典顺序
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    '''处理训练and验证集的问题和答案'''
    questions = utils.path_for(trainval=True, question=True)
    answers = utils.path_for(trainval=True, answer=True)
    # print(questions)   
    # print(answers)
    # /home/gaoling106023/Projects/VQA_data/VQA_V2/Annotations_and_questions/v2_OpenEnded_mscoco_trainval2014_questions.json
    # /home/gaoling106023/Projects/VQA_data/VQA_V2/Annotations_and_questions/v2_mscoco_trainval2014_annotations.json 

    '''处理测试集的问题和答案（无答案）'''
    # questions = utils.path_for(test=True, question=True)
    # answers = utils.path_for(test=True, answer=True)
    # print(questions)
    # print(answers) 
    # /home/gaoling106023/Projects/VQA_data/VQA_V2/Annotations_and_questions/v2_OpenEnded_mscoco_test2015_questions.json
    # /home/gaoling106023/Projects/VQA_data/VQA_V2/Annotations_and_questions/v2_mscoco_val2014_annotations.json

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    # data.prepare_questions:
    # Tokenize and normalize questions from a given question json in the usual VQA format.
    # 用常用的VQA格式,标记和规范化给定question json文件中的 questions
    questions = list(data.prepare_questions(questions))
    answers = list(data.prepare_answers(answers))
    # print(len(questions))# 447793
    # print(len(answers))# 214354
    print('data.prepare_questions successfully!')
    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)
    # print(len(question_vocab))# 12492
    # print(len(answer_vocab))# 3129

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
   
    # '/home/gaoling106023/Projects/VQA2.0-Recent-Approachs-2018/data/vocab.json'
    print(config.vocabulary_path)
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
