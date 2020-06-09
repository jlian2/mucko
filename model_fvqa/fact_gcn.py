import dgl
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import dgl.function as fn


class FactGCN(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, img_dim):
        super(FactGCN, self).__init__()
        self.gcn1 = FastGCNLayer(config, in_dims, out_dims, img_att_proj_dim,
                                 img_dim)

    def forward(self, fbg, ibg):
        fbg = self.gcn1(fbg, ibg)
        return fbg


class FastGCNLayer(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, img_dim):
        super(FastGCNLayer, self).__init__()
        self.config = config
        self.cross_att_fact_proj = nn.Linear(in_dims, img_att_proj_dim)
        self.cross_att_img_proj = nn.Linear(img_dim, img_att_proj_dim)
        self.att_proj = nn.Linear(img_att_proj_dim, 1)
        self.node_fc = nn.Linear(in_dims, out_dims)
        self.img_fc = nn.Linear(img_dim, out_dims)
        self.apply_fc = nn.Linear(2 * out_dims, out_dims)

    def forward(self, fbg, ibg):
        self.img_batch_graph = ibg
        fbg.apply_nodes(func=self.apply_node)
        fbg.update_all(message_func=fn.copy_src(src='h', out='m'),
                       reduce_func=self.reduce)
        return fbg


    def apply_node(self, nodes):
        batch_idx = nodes.data['batch']
        print(
            '=====================================================batch_idex',
            type(batch_idx), batch_idx.shape)
        img_node_ids = filter_img_node(self.img_batch_graph, 'batch',
                                       batch_idx)
        img_features = self.img_batch_graph.nodes[img_node_ids].data[
            'h']  # (36,2048)
        img_proj = self.cross_att_img_proj(img_features)  # (36,512)

        node_proj = self.cross_att_fact_proj(nodes.ndata['h'])  # (512,)
        node_proj = node_proj.unsqueeze(0).repeat(36, 1)  # (36,515)
        node_img_proj = torch.tanh(node_proj + img_proj)  # (36,512)
        att_value = self.att_proj(node_img_proj)  # (36,1)
        att_value = F.softmax(att_value, dim=0)  # (36,1)
        img = torch.matmul(att_value.t(), img_features)  # (1,2048)
        h = self.node_fc(nodes.data['h'])
        return {'img': img, 'h': h}




    def reduce(self, nodes):
        neigh_msg = torch.sum(nodes.mailbox['m'], dim=1)  # shape(out_dim)
        img_msg = self.img_fc(nodes.data['img'])  # out_dim
        msg = neigh_msg + img_msg

        h = nodes.data['h']  
        h = torch.cat([msg, h], dim=1)  
        h = F.relu(self.apply_fc(h))  
        return {'h': h}


def filter_img_node(img_batch_graph, attribute, value):
    print(
        '===================================img_batch_graph.ndata[attribute]',
        img_batch_graph.ndata[attribute].shape)
    print('========================================value', type(value),
          value.shape)
    mask = (img_batch_graph.ndata[attribute] == value).squeeze()
    return torch.masked_select(img_batch_graph.nodes(), mask)
