import dgl
import torch

import torch.nn.functional as F
from torch import nn
import numpy as np
import dgl.function as fn


class FactGCN2(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim, fact_dim,
                 gate_dim):
        super(FactGCN2, self).__init__()
        self.gcn = FactGCNLayer2(config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim,
                                 fact_dim,
                                 gate_dim)
        # self.mlp = nn.Sequential(nn.Linear(out_dims, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, fact_batch_graph, img_batch_graph, sem_batch_graph):
        fact_graphs = dgl.unbatch(fact_batch_graph)
        img_graphs = dgl.unbatch(img_batch_graph)
        sem_graphs = dgl.unbatch(sem_batch_graph)
        num_graph = len(fact_graphs)
        new_fact_graphs = []
        for i in range(num_graph):
            fact_graph = fact_graphs[i]
            img_graph = img_graphs[i]
            sem_graph = sem_graphs[i]
         
            fact_graph = self.gcn(fact_graph, img_graph, sem_graph)

            new_fact_graphs.append(fact_graph)
        return dgl.batch(new_fact_graphs)


class FactGCNLayer2(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim, fact_dim,
                 gate_dim):
        super(FactGCNLayer2, self).__init__()
        self.config = config

        self.cross_img_att_fact_proj = nn.Linear(in_dims, img_att_proj_dim)
        self.cross_img_att_img_proj = nn.Linear(img_dim, img_att_proj_dim)
        self.img_att_proj = nn.Linear(img_att_proj_dim, 1)
        
        self.cross_sem_att_fact_proj = nn.Linear(in_dims, sem_att_proj_dim)
        self.cross_sem_att_node_proj = nn.Linear(sem_dim, sem_att_proj_dim)
        self.sem_att_proj = nn.Linear(sem_att_proj_dim, 1)
        
        self.node_fc = nn.Linear(in_dims, in_dims)
        self.apply_fc = nn.Linear( in_dims, fact_dim)
        self.gate_dim=gate_dim

        # gate
        self.img_gate_fc = nn.Linear(img_dim, gate_dim)
        self.sem_gate = nn.Linear(sem_dim, gate_dim)
        self.fact_gate = nn.Linear(out_dims, gate_dim)
        self.gate_fc = nn.Linear(3 * gate_dim, 3 * gate_dim)
        self.out_fc = nn.Linear(3 * gate_dim, out_dims)

    def forward(self, fact_graph, img_graph, sem_graph):
        self.img_graph = img_graph
        self.fact_graph = fact_graph
        self.sem_graph = sem_graph

        fact_graph.apply_nodes(func=self.apply_node)
        fact_graph.update_all(message_func=fn.copy_src(src='h', out='m'),
                              reduce_func=self.reduce)
        return fact_graph


    def apply_node(self, nodes):
    

        node_features = nodes.data['h']


        img_features = self.img_graph.ndata['h']  # (36,2048)
        img_proj = self.cross_img_att_img_proj(img_features)  # (36,512)
        node_proj = self.cross_img_att_fact_proj(node_features)  # (num,512)
        node_proj = node_proj.repeat(1, 36, 1)  # (num,36,512)
        img_proj = img_proj.repeat(
            self.fact_graph.number_of_nodes(), 1, 1)
        node_img_proj = torch.tanh(node_proj + img_proj)  # (num,36,512)
        img_att_value = self.img_att_proj(node_img_proj).squeeze()  # (num,36)
        img_att_value = F.softmax(img_att_value, dim=1)  # (num,36)
        img = torch.matmul(img_att_value, img_features)  # (num,2048)



        sem_features = self.sem_graph.ndata['h']  # (n,2048)
        sem_num_nodes = self.sem_graph.number_of_nodes()
        sem_proj = self.cross_sem_att_node_proj(sem_features)  # (n,512)
        node_proj = self.cross_sem_att_fact_proj(node_features)  # (n,512)
        node_proj = node_proj.unsqueeze(1).repeat(1, sem_num_nodes, 1)  # (num,n,512)
        sem_proj = sem_proj.unsqueeze(0).repeat(
            self.fact_graph.number_of_nodes(), 1, 1)
        node_sem_proj = torch.tanh(node_proj + sem_proj)  # (num,36,512)
        sem_att_value = self.sem_att_proj(node_sem_proj).squeeze()  # (num,36)
        sem_att_value = F.softmax(sem_att_value, dim=1)  # (num,36)
        sem = torch.matmul(sem_att_value, sem_features)  # (num,2048)


        h = self.node_fc(node_features)
        return {'img': img, 'sem': sem, 'h': h}




    def reduce(self, nodes):
        neigh_msg = torch.mean(nodes.mailbox['m'], dim=1)  # shape(out_dim)
  

        h = nodes.data['h']  # shape(gate_dim)
        h = torch.cat([neigh_msg, h], dim=1)  # shape(2*outdim)
        h = nodes.data['att'] * F.relu(self.apply_fc(h))  # shape(gate_dim)


        img = self.img_gate_fc(nodes.data['img'])
        sem = self.sem_gate(nodes.data['sem'])
        fact = self.fact_gate(h)

        gate = torch.sigmoid(self.gate_fc(torch.cat([fact, img, sem], dim=1)))
        w1 = gate[:self.gate_dim].sum()/gate.sum()
        w2 =gate[self.gate_dim:self.gate_dim*2].sum()/gate.sum()
        w3 =gate[self.gate_dim*2:].sum()/gate.sum()

        h = self.out_fc(gate * torch.cat([fact, img, sem], dim=1))
        return {'h': h}
