import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as numpy
from util.dynamic_rnn import DynamicRNN
from model_fvqa.img_gcn import ImageGCN
from model_fvqa.fact_gcn import FactGCN
from model_fvqa.fact_gcn2 import FactGCN2

import dgl
import networkx as nx
import numpy as np


class CMGCNnet(nn.Module):
    def __init__(self, config, que_vocabulary, glove, device):
        '''
        :param config: 配置参数
        :param que_vocabulary: 字典 word 2 index
        :param glove: (voc_size,embed_size)
        '''
        super(CMGCNnet, self).__init__()
        self.config = config
        self.device = device

        self.que_glove_embed = nn.Embedding(len(que_vocabulary), config['model']['glove_embedding_size'])

        self.que_glove_embed.weight.data = glove

        self.que_glove_embed.weight.requires_grad = False


        self.ques_rnn = nn.LSTM(config['model']['glove_embedding_size'],
                                config['model']['lstm_hidden_size'],
                                config['model']['lstm_num_layers'],
                                batch_first=True,
                                dropout=config['model']['dropout'])
        self.ques_rnn=DynamicRNN(self.ques_rnn)


        self.node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)


        self.rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.rel_att_proj_rel = nn.Linear(
            config['model']['relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)


        self.img_gcn1 = ImageGCN(config,
                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['relation_dims'])

        self.fact_gcn1 = FactGCN(
            config,
            in_dims=config['model']['fact_gcn1_feature_dim'],
            out_dims=config['model']['fact_gcn1_out_dim'],
            img_att_proj_dim=config['model']['fact_gcn1_att_proj_dim'],
            img_dim=config['model']['image_gcn1_out_dim'])

        self.new_fact_gcn1 = FactGCN2(
            config,
            in_dims=config['model']['fact_gcn1_feature_dim'],
            out_dims=config['model']['fact_gcn1_out_dim'],
            img_att_proj_dim=config['model']['fact_gcn1_att_proj_dim'],
            img_dim=config['model']['image_gcn1_out_dim'],
            que_dim=config['model']['lstm_hidden_size'])


        self.img_gcn2 = ImageGCN(config,
                                 in_dim=config['model']['image_gcn1_out_dim'],
                                 out_dim=config['model']['image_gcn2_out_dim'],
                                 rel_dim=config['model']['relation_dims'])

        self.fact_gcn2 = FactGCN(
            config,
            in_dims=config['model']['fact_gcn1_out_dim'],
            out_dims=config['model']['fact_gcn2_out_dim'],
            img_att_proj_dim=config['model']['fact_gcn2_att_proj_dim'],
            img_dim=config['model']['image_gcn2_out_dim'])

        self.new_fact_gcn2 = FactGCN2(
            config,
            in_dims=config['model']['fact_gcn1_out_dim'],
            out_dims=config['model']['fact_gcn2_out_dim'],
            img_att_proj_dim=config['model']['fact_gcn2_att_proj_dim'],
            img_dim=config['model']['image_gcn2_out_dim'],
            que_dim=config['model']['lstm_hidden_size'])



    def forward(self, batch):

        # ======================================================================================
        #                                    数据处理
        # ======================================================================================
        batch_size = len(batch['facts_num_nodes_list'])

        images = batch['features_list']  # [(36,2048)]
        images = torch.stack(images).to(self.device)  # [batch,36,2048]

        img_relations = batch['img_relations_list']
        img_relations = torch.stack(img_relations).to(self.device)  # shape (batch,36,36,7) 暂定7维


        questions = batch['question_list']  # list((max_length,))
        questions = torch.stack(questions).long().to(self.device)  # [batch,max_length]
        questions_len_list = batch['question_length_list']
        questions_len_list = torch.tensor(batch['question_length_list']).long().to(self.device)





        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list']).long().to(self.device)
        facts_features_list = batch['facts_features_list']
        facts_features_list = [torch.Tensor(features).to(self.device)
                               for features in facts_features_list
                               ]
        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [
            torch.Tensor(e1ids).long().to(self.device)
            for e1ids in facts_e1ids_list
        ]
        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [
            torch.tensor(e2ids).long().to(self.device)
            for e2ids in facts_e2ids_list
        ]
        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [
            torch.tensor(answer).long().to(self.device)
            for answer in facts_answer_list
        ]

        ques_embed = self.que_glove_embed(questions).float()  
     
        _, (ques_embed, _)=self.ques_rnn(ques_embed,questions_len_list)
   
        node_att_proj_ques_embed = self.node_att_proj_ques(ques_embed)  
        node_att_proj_img_embed = self.node_att_proj_img(images)  

        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1],
                                                                                1)  
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        node_att_values = self.node_att_value(node_att_proj_img_sum_ques).squeeze()  
        node_att_values = F.softmax(node_att_values, dim=-1)  

        node_att_values = node_att_values.unsqueeze(-1).repeat(
            1, 1, self.config['model']['img_feature_size'])


        images = node_att_values * images  # (b,36)*(b,36,2048)

       
        rel_att_proj_ques_embed = self.rel_att_proj_ques(
            ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.rel_att_proj_rel(
            img_relations)  # shape(batch,36,36,128)

        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        rel_att_values = self.rel_att_value(
            rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)
        rel_att_values_2 = rel_att_values.unsqueeze(-1).repeat(
            1, 1, 1, self.config['model']['relation_dims'])


        img_relations = rel_att_values_2 * img_relations  # (batch,36,36,7)

  
        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()
            # add nodes
            g.add_nodes(36)
            # add node features
            g.ndata['h'] = images[i]
            g.ndata['batch'] = torch.full([36, 1], i)
            # add edges
            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)
            # add edge features
            g.edata['rel'] = img_relations[i].view(
                36 * 36,
                self.config['model']['relation_dims'])  # shape(36*36,7)
            g.edata['att'] = rel_att_values[i].view(36 * 36,
                                                    1)  # shape(36*36,1)
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)

       
        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(fact_num_nodes_list[i])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i)
            graph.ndata['answer'] = facts_answer_list[i]

            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)


        image_batch_graph = self.img_gcn1(image_batch_graph)
        
        fact_batch_graph = self.new_fact_gcn1(fact_batch_graph,
                                              image_batch_graph,
                                              ques_embed=ques_embed)
        fact_batch_graph.ndata['h'] = F.relu(fact_batch_graph.ndata['h'])
      
        image_batch_graph = self.img_gcn2(image_batch_graph)

        fact_batch_graph = self.new_fact_gcn2(
            fact_batch_graph, image_batch_graph,
            ques_embed=ques_embed)  # 每个节点1 个特征
        fact_batch_graph.ndata['h'] = torch.sigmoid(fact_batch_graph.ndata['h'])
        fact_batch_graph.ndata['h'] = torch.softmax(
            fact_batch_graph.ndata['h'], dim=0)
        return fact_batch_graph
