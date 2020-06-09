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


class FvqaTestDataset(Dataset):
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


        with open(config['dataset']['test']['test_qids'], 'r') as f:
            self.test_qids = json.load(f)

        with open(config['dataset']['test']['test_questions'], 'r') as f:
            self.test_questions = json.load(f)

        with open(config['dataset']['test']['test_answers'], 'r') as f:
            self.test_answers = json.load(f)

        with open(config['dataset']['test']['test_gt_facts'], 'r') as f:
            self.test_gt_facts = json.load(f)

        with open(config['dataset']['test']['test_captions'], 'r') as f:
            self.test_captions = json.load(f)

        with open(config['dataset']['test']['test_semantic_graph'], 'rb') as f:
            self.semantic_graphs = pickle.load(f, encoding='iso-8859-1')

        with open(config['dataset']['test']['test_labels'], 'r') as f:
            self.test_labels = json.load(f)

        with open(config['dataset']['test']['test_whs'], 'r') as f:
            self.test_whs = json.load(f)


        with open(config['dataset']['test']['test_top100_facts_graph'],
                  'rb') as f:
            self.test_top_facts = pickle.load(f, encoding='iso-8859-1')

        self.test_features = np.load(
            config['dataset']['test']['test_features'])

        self.test_bboxes = np.load(config['dataset']['test']['test_bboxes'])

        if overfit:
            self.test_qids = self.test_qids[:100]
            self.test_questions = self.test_questions[:100]
            self.test_answers = self.test_answers[:100]
            self.test_gt_facts = self.test_gt_facts[:100]

            self.test_top_facts = self.test_top_facts[:100]
            self.test_captions = self.test_captions[:100]
            self.test_bboxes = self.test_bboxes[:100]
            self.test_features = self.test_features[:100]
            self.test_whs = self.test_whs[:100]
            self.semantic_graphs=self.semantic_graphs[:100]

    def __getitem__(self, index):
        test_id = self.test_qids[index]
        test_question = self.test_questions[index]
        test_answer = self.test_answers[index]
        test_gt_fact = self.test_gt_facts[index]

        test_top_facts = self.test_top_facts[index]
        test_captions = self.test_captions[index]
        test_bboxes = self.test_bboxes[index]  # (36,4)
        test_features = torch.tensor(self.test_features[index])
        semantic_graph=self.semantic_graphs[index]


        w = self.test_whs[index][0]
        h = self.test_whs[index][1]
        img_relations = torch.zeros(36, 36, 7)

        for i in range(36):
            for j in range(36):
                xi = test_bboxes[i][0]
                yi = test_bboxes[i][1]
                wi = test_bboxes[i][2]
                hi = test_bboxes[i][3]
                xj = test_bboxes[j][0]
                yj = test_bboxes[j][1]
                wj = test_bboxes[j][2]
                hj = test_bboxes[j][3]

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
            test_features = normalize(test_features, dim=0, p=2)

        # 对 question 转化为 index
        question_length = len(test_question.split())

        q_indices = self.que_vocabulary.to_indices(test_question.split())
        test_question = self.pad_sequences(q_indices)

        item = {}

        item['id'] = test_id  # scalar
        item['question'] = test_question  # (max_len,)
        
        item['question_length'] = question_length  # scalar

        item['features'] = test_features  # (36,2048)
        
        item['img_relations'] = img_relations  # (36,36,7)

        item['facts_num_nodes'] = len(test_top_facts['nodes'])  # scalar

        item['facts_features'] = test_top_facts['features']  # (num,2048)
        item['facts_e1ids'] = test_top_facts['e1ids']  # (num_edges,)
        item['facts_e2ids'] = test_top_facts['e2ids']  # (num_edges,)

        item['facts_answer'] = test_top_facts['answer']
        item['facts_answer_id'] = test_top_facts['answer_id']  # scalar
        item['semantic_num_nodes']=len(semantic_graph['nodes'])
        item['semantic_n_features']=semantic_graph['n_features']
        item['semantic_e1ids']=semantic_graph['e1ids']
        item['semantic_e2ids']=semantic_graph['e2ids']
        item['semantic_e_features']=semantic_graph['e_features']
        return item

    def __len__(self):
        return len(self.test_qids)

    def pad_sequences(self, sequence):

        sequence = sequence[:self.config['dataset']['max_sequence_lengtn']]

        padding = np.zeros(self.config['dataset']['max_sequence_lengtn'])
        padding[:len(sequence)] = np.array(sequence)
        return torch.tensor(padding)

