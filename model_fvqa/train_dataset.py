from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import json
import numpy as np
import pickle
import torch
from util.vocabulary import Vocabulary
from torch.utils.data import DataLoader
import dgl
from math import sqrt, atan2
import yaml

FIELDNAMES = [
    'image_id', 'image_w', 'image_h', 'num_boxes', 'labels', 'attr_labels',
    'boxes', 'features'
]


class FvqaTrainDataset(Dataset):
    def __init__(self, config, overfit=False):
        super().__init__()

        self.que_vocabulary = Vocabulary(config["dataset"]["word2id_path"])

        self.image_features = []
        self.image_boxes = []
        self.image_labels = []
        self.image_captions = []

        self.questions = []
        self.questions_ids = []
        self.questions_vecs = []
        self.answers = []
        self.config = config

        # 问题的 id
        with open(config['dataset']['train']['train_qids'], 'r') as f:
            self.train_qids = json.load(f)
        # 问题
        with open(config['dataset']['train']['train_questions'], 'r') as f:
            self.train_questions = json.load(f)
        # 答案
        with open(config['dataset']['train']['train_answers'], 'r') as f:
            self.train_answers = json.load(f)
        # gt fact
        with open(config['dataset']['train']['train_gt_facts'], 'r') as f:
            self.train_gt_facts = json.load(f)
        # caption
        with open(config['dataset']['train']['train_captions'], 'r') as f:
            self.train_captions = json.load(f)
        # semantic graph
        with open(config['dataset']['train']['train_semantic_graph'], 'rb') as f:
            self.semantic_graphs = pickle.load(f, encoding='iso-8859-1')
        # 图像bbox 对应的 label
        with open(config['dataset']['train']['train_labels'], 'r') as f:
            self.train_labels = json.load(f)
        # 图像的长宽
        with open(config['dataset']['train']['train_whs'], 'r') as f:
            self.train_whs = json.load(f)
        # 抽取到的 facts
        # with open(config['dataset']['train']['train_facts_graph'], 'rb') as f:
        #     self.train_facts = pickle.load(f, encoding='iso-8859-1')
        with open(config['dataset']['train']['train100_facts_graph'],
                  'rb') as f:
            self.train_top_facts = pickle.load(f, encoding='iso-8859-1')
        # 图像 bbox 的特征
        self.train_features = np.load(
            config['dataset']['train']['train_features'])
        # 图像的 bbox 几何信息
        self.train_bboxes = np.load(config['dataset']['train']['train_bboxes'])

        if overfit:
            self.train_qids = self.train_qids[:100]
            self.train_questions = self.train_questions[:100]
            self.train_answers = self.train_answers[:100]
            self.train_gt_facts = self.train_gt_facts[:100]
            # self.train_facts = self.train_facts[:100]
            self.train_top_facts = self.train_top_facts[:100]
            self.train_captions = self.train_captions[:100]
            self.train_bboxes = self.train_bboxes[:100]
            self.train_features = self.train_features[:100]
            self.train_whs = self.train_whs[:100]
            self.semantic_graphs=self.semantic_graphs[:100]

    def __getitem__(self, index):
        train_id = self.train_qids[index]
        train_question = self.train_questions[index]
        train_answer = self.train_answers[index]
        train_gt_fact = self.train_gt_facts[index]
        # train_facts = self.train_facts[index]
        train_top_facts = self.train_top_facts[index]
        train_captions = self.train_captions[index]
        train_bboxes = self.train_bboxes[index]  # (36,4)
        train_features = torch.tensor(self.train_features[index])
        semantic_graph=self.semantic_graphs[index]

        # cal relation info （36,36,7）
        w = self.train_whs[index][0]
        h = self.train_whs[index][1]
        img_relations = torch.zeros(36, 36, 7)

        for i in range(36):
            for j in range(36):
                xi = train_bboxes[i][0]
                yi = train_bboxes[i][1]
                wi = train_bboxes[i][2]
                hi = train_bboxes[i][3]
                xj = train_bboxes[j][0]
                yj = train_bboxes[j][1]
                wj = train_bboxes[j][2]
                hj = train_bboxes[j][3]

                r1 = (xj - xi) / (wi * hi) ** 0.5
                r2 = (yj - yi) / (wi * hi) ** 0.5
                r3 = wj / wi
                r4 = hj / hi
                r5 = (wj * hj) / wi * hi
                r6 = sqrt((xj - xi) ** 2 + (yj - yi) ** 2) / sqrt(w ** 2 + h ** 2)
                r7 = atan2(yj - yi, xj - xi)

                rel = torch.tensor([r1, r2, r3, r4, r5, r6, r7])
                img_relations[i][j] = rel

        # 归一化
        if self.config['dataset']["img_norm"]:
            train_features = normalize(train_features, dim=0, p=2)

        # 对 question 转化为 index
        question_length = len(train_question.split())
        # train_question=self.que_vocabulary.to_indices(train_question.split())
        q_indices = self.que_vocabulary.to_indices(train_question.split())
        train_question = self.pad_sequences(q_indices)

        item = {}
        # question
        item['id'] = train_id  # scalar
        item['question'] = train_question  # (max_len,)
        # item['gt_fact'] = train_gt_fact  # [e1,e2,r]
        # item['answer'] = train_answer  # string
        item['question_length'] = question_length  # scalar
        # image
        item['features'] = train_features  # (36,2048)
        # item['bboxes'] = train_bboxes  # (36,4)
        # item['captions'] = train_captions  # ()
        item['img_relations'] = img_relations  # (36,36,7)
        # fact graph
        item['facts_num_nodes'] = len(train_top_facts['nodes'])  # scalar
        # item['facts_nodes'] = train_top_facts['nodes']  # (num_nodes,)
        item['facts_features'] = train_top_facts['features']  # (num,2048)
        item['facts_e1ids'] = train_top_facts['e1ids']  # (num_edges,)
        item['facts_e2ids'] = train_top_facts['e2ids']  # (num_edges,)
        # (num_nodes,)  one-hot
        item['facts_answer'] = train_top_facts['answer']
        item['facts_answer_id'] = train_top_facts['answer_id']  # scalar
        item['semantic_num_nodes']=len(semantic_graph['nodes'])
        item['semantic_n_features']=semantic_graph['n_features']
        item['semantic_e1ids']=semantic_graph['e1ids']
        item['semantic_e2ids']=semantic_graph['e2ids']
        item['semantic_e_features']=semantic_graph['e_features']
        return item

    def __len__(self):
        return len(self.train_qids)

    def pad_sequences(self, sequence):
        # 超出的裁剪
        sequence = sequence[:self.config['dataset']['max_sequence_lengtn']]
        # 没超出的padding
        padding = np.zeros(self.config['dataset']['max_sequence_lengtn'])
        padding[:len(sequence)] = np.array(sequence)
        return torch.tensor(padding)


