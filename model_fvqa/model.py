import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as numpy
from util.dynamic_rnn import DynamicRNN
from model_fvqa.img_gcn import ImageGCN
from model_fvqa.semantic_gcn import SemanticGCN
from model_fvqa.fact_gcn import FactGCN
from model_fvqa.fact_gcn2 import FactGCN2

import dgl
import networkx as nx
import numpy as np


class CMGCNnet(nn.Module):
    def __init__(self, config, que_vocabulary, glove, device):
      
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
        self.ques_rnn = DynamicRNN(self.ques_rnn)


        self.vis_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)


        self.vis_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_proj_rel = nn.Linear(
            config['model']['relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)


        self.sem_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_proj_sem = nn.Linear(
            config['model']['sem_node_dims'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_value = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims'], 1)


        self.sem_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_proj_rel = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)


        self.fact_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_proj_node = nn.Linear(
            config['model']['fact_node_dims'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_value = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims'], 1)


        self.img_gcn1 = ImageGCN(config,
                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['relation_dims'])


        self.sem_gcn1 = SemanticGCN(config,
                                    in_dim=config['model']['sem_node_dims'],
                                    out_dim=config['model']['semantic_gcn1_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])
     

        self.new_fact_gcn1 = FactGCN2(
            config,
            in_dims=config['model']['fact_gcn1_feature_dim'],
            out_dims=config['model']['fact_gcn1_out_dim'],
            img_att_proj_dim=config['model']['fact_gcn1_img_att_proj_dim'],
            sem_att_proj_dim=config['model']['fact_gcn1_sem_att_proj_dim'],
            img_dim=config['model']['image_gcn1_out_dim'],
            sem_dim=config['model']['semantic_gcn1_out_dim'],
            fact_dim=config['model']['fact_gcn1_fact_dim'],
            gate_dim=config['model']['fact_gcn1_gate_dim'])

  

        self.mlp = nn.Sequential(
            nn.Linear(config['model']['fact_gcn1_out_dim'] + config['model']['lstm_hidden_size'], 1024), nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid())

    def forward(self, batch):

        
        batch_size = len(batch['facts_num_nodes_list'])

        images = batch['features_list']  
        images = torch.stack(images)  

        img_relations = batch['img_relations_list']
        img_relations = torch.stack(img_relations)  


        questions = batch['question_list'] 
        questions = torch.stack(questions) 
        questions_len_list = batch['question_length_list']
        questions_len_list = torch.tensor(batch['question_length_list'])


        semantic_num_nodes_list = torch.Tensor(batch['semantic_num_nodes_list'])
        semantic_n_features_list = batch['semantic_n_features_list']
        semantic_n_features_list = [torch.Tensor(features)
                                    for features in semantic_n_features_list
                                    ]
        semantic_e1ids_list = batch['semantic_e1ids_list']
        semantic_e1ids_list = [
            torch.Tensor(e1ids)
            for e1ids in semantic_e1ids_list
        ]
        semantic_e2ids_list = batch['semantic_e2ids_list']
        semantic_e2ids_list = [
            torch.Tensor(e1ids)
            for e1ids in semantic_e2ids_list
        ]
        semantic_e_features_list = batch['semantic_e_features_list']
        semantic_e_features_list = [torch.Tensor(features)
                                    for features in semantic_e_features_list
                                    ]

        # fact graph
        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list'])
        facts_features_list = batch['facts_features_list']
        facts_features_list = [torch.Tensor(features)
                               for features in facts_features_list
                               ]
        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [
            torch.Tensor(e1ids)
            for e1ids in facts_e1ids_list
        ]
        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [
            torch.tensor(e2ids)
            for e2ids in facts_e2ids_list
        ]
        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [
            torch.tensor(answer)
            for answer in facts_answer_list
        ]
       
        ques_embed = self.que_glove_embed(questions).float()  # shape (batch,max_length,300)
        
        _, (ques_embed, _) = self.ques_rnn(ques_embed, questions_len_list)  # qes_embed shape=(batch,hidden_size)

       
        node_att_proj_ques_embed = self.vis_node_att_proj_ques(ques_embed)  # shape (batch,proj_size)
        node_att_proj_img_embed = self.vis_node_att_proj_img(images)  # shape (batch,36,proj_size)

        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1],
                                                                                1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)


        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques(ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel(img_relations)  # shape(batch,36,36,128)

        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        for i in range(batch_size):

            num_node = semantic_num_nodes_list[i]  # n
            sem_node_features = semantic_n_features_list[i]  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques(q_embed)  # shape (n,p)
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem(sem_node_features)  # shape (n,p)
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed)  # shape (n,p)
            sem_node_att_values = self.sem_node_att_value(sem_node_att_proj_sem_sum_ques)  # shape(n,1)
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  # shape(n,1)

            sem_node_att_val_list.append(sem_node_att_values)


            num_edge = semantic_e_features_list[i].shape[0]  # n
            sem_edge_features = semantic_e_features_list[i]  # (n,300)
            qq_embed = ques_embed[i]  # (512)
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  # (n,512)
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques(qq_embed)  # shape (n,p)
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel(sem_edge_features)  # shape (n,p)
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  # shape (n,p)
            sem_rel_att_values = self.sem_rel_att_value(sem_rel_att_proj_rel_sum_ques)  # shape(n,1)
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  # shape(n,1)

            sem_edge_att_val_list.append(sem_rel_att_values)



        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()

            g.add_nodes(36)

            g.ndata['h'] = images[i]
            g.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)
            g.ndata['batch'] = torch.full([36, 1], i)

            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)

            g.edata['rel'] = img_relations[i].view(36 * 36, self.config['model']['relation_dims'])  
            g.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)  
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)

     
        semantic_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(semantic_num_nodes_list[i])
            graph.add_edges(semantic_e1ids_list[i], semantic_e2ids_list[i])
            graph.ndata['h'] = semantic_n_features_list[i]
            graph.ndata['att'] = sem_node_att_val_list[i]
            graph.edata['r'] = semantic_e_features_list[i]
            graph.edata['att'] = sem_edge_att_val_list[i]
            semantic_graphs.append(graph)
        semantic_batch_graph = dgl.batch(semantic_graphs)

       
        fact_node_att_values_list = []
        for i in range(batch_size):
            num_node = fact_num_nodes_list[i]  # n
            fact_node_features = facts_features_list[i]  # (n,1024)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)

            fact_node_att_values_list.append(fact_node_att_values)

       
        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(fact_num_nodes_list[i])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]
            graph.ndata['att'] = fact_node_att_values_list[i]
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i)
            graph.ndata['answer'] = facts_answer_list[i]
            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)

       
        image_batch_graph = self.img_gcn1(image_batch_graph)

        semantic_batch_graph = self.sem_gcn1(semantic_batch_graph)

        fact_batch_graph = self.new_fact_gcn1(fact_batch_graph, image_batch_graph, semantic_batch_graph)


        fact_graphs = dgl.unbatch(fact_batch_graph)
        new_fact_graphs = []
        for i, fact_graph in enumerate(fact_graphs):
            num_nodes = fact_graph.number_of_nodes()
            q_embed = ques_embed[i]
            q_embed=q_embed.unsqueeze(0).repeat(num_nodes, 1)
            fact_graph.ndata['h'] = torch.cat([fact_graph.ndata['h'], q_embed], dim=1)
            new_fact_graphs.append(fact_graph)
        fact_batch_graph = dgl.batch(new_fact_graphs)

        fact_batch_graph.ndata['h'] = self.mlp(fact_batch_graph.ndata['h'])

       
        return fact_batch_graph
