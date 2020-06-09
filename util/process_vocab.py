import os
import json
import itertools
import pickle
import re
from collections import Counter
import yaml
from tqdm import tqdm
from bert_serving.client import BertClient

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')
config = yaml.load(open('../exp_fvqa/exp.yml'))


def get_all_que_ans(file):
    '''
    将 answer 和 question 分离，分别得到 all_questions.txt和 all_answers.txt
    :param file:
    :return:
    '''
    question_list = []
    answer_list = []

    with open(file, 'r') as f:
        raw_data = json.load(f)
        for id, item in raw_data.items():
            question = item["question"].strip().lower().replace('?', '')
            question = _special_chars.sub('', question).strip()
            question_list.append(question)

            answer = item["answer"].lower().replace('a ', '')
            answer = _special_chars.sub('', answer).strip()
            answer_list.append(answer)

    with open('fvqa/exp_data/data/all_questions.txt',
              'w') as f:
        for question in question_list:
            f.write(question + '\n')

    with open('fvqa/exp_data/data/all_answers.txt',
              'w') as f:
        for answer in answer_list:
            f.write(answer + '\n')

    print('finished!!!')


def extract_question_vocab(file, top_k=None, start=0):
    question_list = []
    with open(file, 'r') as f:
        raw_data = json.load(f)
        for id, item in raw_data.items():
            question = item["question"]
            question_list.append(question.split())

    max_len = 0
    for q in question_list:
        if len(q) > max_len:
            max_len = len(q)

    print('max len' + str(max_len))

    all_tokens = itertools.chain.from_iterable(question_list)
    counter = Counter(all_tokens)

    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()

    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}

    with open('fvqa/exp_data/data/vocab_question.json',
              'w') as f:
        json.dump(vocab, f)

    print('vocab_question.json finished!!!')


def extract_answer_vocab(file, top_k=None, start=0):
    answer_list = []
    with open(file, 'r') as f:
        raw_data = json.load(f)
        for id, item in raw_data.items():
            id = int(id)
            answer = item["answer"].lower().replace('a ', '').strip()
            answer = _special_chars.sub('', answer).strip()
            answer_list.append(answer)

    counter = Counter(answer_list)

    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()

    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}

    with open('fvqa/exp_data/data/vocab_answer.json',
              'w') as f:
        json.dump(vocab, f)

    print('vocab_answer.json finished!!!')


def get_word2id():
    word2id_file = 'fvqa/exp_data/data/word2id_question.json'
    f = open(word2id_file, 'r')
    word2id = json.load(f)
    f.close()
    return word2id


def get_id2word():
    id2word_file = 'fvqa/exp_data/data/id2word_question.json'
    f = open(id2word_file, 'r')
    id2word = json.load(f)
    print('ok')


def fact_vob():
    train_fact_path = config['dataset']['train']['train_facts']
    test_fact_path = config['dataset']['test']['test_facts']
    dict = set()
    max_len = 0
    with open(train_fact_path, 'r') as f:
        train_fact = json.load(f)

    for item in tqdm(train_fact):
        for fact in item:
            e1 = fact[0].split()
            e2 = fact[1].split()
            if len(e2) > len(e1) and len(e2) > max_len:
                max_len = len(e2)
            dict = dict.union(set(e1 + e2))
    fact_word2id = {}
    fact_id2word = {}
    for i, word in enumerate(dict):
        fact_id2word[i] = word
        fact_word2id[word] = i

    with open('fvqa/exp_data/data/id2word_fact.json', 'w') as f:
        json.dump(fact_id2word, f)

    with open('fvqa/exp_data/data/word2id_fact.json', 'w') as f:
        json.dump(fact_word2id, f)
    print('finished')
    print('max_len:', max_len)





if __name__ == "__main__":
    extract_question_vocab('fvqa/exp_data/data/all_qs_rel_dict_release.json')
