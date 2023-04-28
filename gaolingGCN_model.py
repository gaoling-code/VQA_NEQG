##########################
# Implementation of Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
# Paper Link: https://arxiv.org/abs/1707.07998
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import word_embedding

from reuse_modules import Fusion, FCNet

class Net(nn.Module):
    def __init__(self, words_list):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features    # 2048
        glimpses = 2

        self.text = word_embedding.TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.0,
        )

        # self.rw = RandomWalk()

        self.GCNnet = GCN(input_size=vision_features,
        hidden_size=vision_features,
        out_size=vision_features,
        # dropout=DROPOUT
        )

        self.attention1 = Attention_Horizontal(
            v_features=vision_features,
            q_features=question_features,
            mid_features=1024,
            glimpses=glimpses,
            drop=0.2,)
        
        self.attention2 = Attention_Vertical(
            v_features=vision_features,
            q_features=question_features,
            mid_features=1024,
            glimpses=glimpses,
            drop=0.2,)
            
        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=config.max_answers,# 3129
            drop=0.5,)

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        '''
        v: visual feature      [batch, num_obj, 2048]
        b: bounding box        [batch, num_obj, 4]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        q = self.text(q, list(q_len.data))  # [batch, 1024]
        if config.v_feat_norm: 
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v) # [batch, num_obj=36, 2048]

        # GCN!
        features_similarities = CosineSimilarity(v)     # Cosine similarity [batch, 36, 36]
        v_head = self.GCNnet(features_similarities, v)

        # v_head = self.rw(features_similarities, v) # [batch=256, num_objs=36, obj_dim=2048]
        
        # 横向attention
        a1 = self.attention1(v_head, q) # [batch=256, num_obj=36, num_glimpse=2]
        v_head1 = apply_attention(v_head.transpose(1,2), a1) # [batch=256, 2048*num_glimpse=2048*2=4096]
        # 纵向attention
        a2 = self.attention2(v_head, q) # [batch=256, 2048, 64]
        v_head2 = apply_attention(v_head, a2) # [batch=256, 36*64=2304]

        v = torch.cat([v_head1, v_head2], 1)# 水平拼接[batch=256, 36*64+2048*2=6400]
        answer = self.classifier(v, q)# [batch=256, 3129]

        return answer 

# 计算 欧几里得距离 和 欧几里得相似性
def OsDistance1(vector1, vector2):
    distance = np.linalg.norm(vector1-vector2)
    return distance
def OsSimilarity(vector1, vector2):
    # vector1:[2048]
    # vector2:[2048]
    distance = vector1 @ vector2
    # distance = np.linalg.norm(vector1-vector2)
    similarity = 1 / (1 + distance)
    return similarity

# Cosine similarity
def CosineSimilarity(vector1):
    # v = torch.rand(256,36,128)
    S = torch.matmul(vector1,vector1.transpose(-1,-2))
    a = torch.diagonal(S,dim1=-2,dim2=-1).unsqueeze(dim=-1).sqrt()
    s = torch.matmul(a,a.transpose(-1,-2))
    similarity = S/s
    similarity = (similarity+1)/2
    return similarity

class RandomWalk(nn.Module):
    def __init__(self):
        super(RandomWalk, self).__init__()

    def forward(self, Association_matrix, v):   # similarity, feature
        """
        s: features_similarities[batch, k, k]
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        alpha = 0.9     # 衡量原始信息与随机游走后的信息的权重
        step = 10        # 随机游走的步数
        probabilistic_transfer_matrix = torch.nn.functional.normalize(Association_matrix, p=1, dim=1)     # 行规一化为概率转换矩阵 [2560, 36, 36]
        probabilistic_transfer_matrix_T = probabilistic_transfer_matrix.transpose(1,2)              # 概率转换矩阵的转置 [2560, 36, 36]

        P0 = probabilistic_transfer_matrix  # 行规一化后的关联矩阵（即：概率转换矩阵）
        P_t = P0
        for t in range(step):
            old_vPk = P_t 
            P_t = (1 - alpha) * probabilistic_transfer_matrix_T @ P_t+ alpha * P0
            # 判断是否吻合
            vd = P_t - old_vPk
            # y = np.linalg.norm(vd)  # 求范数
            # y = torch.norm(vd, p='fro', dim=2, keepdim=False, out=None, dtype=None)  # torch求范数
            y = torch.norm(vd, dim=(1,2))   # torch.Size([batch])
            y_max = torch.max(y)
            if y_max < (1e-06):
                v_head = P_t @ v        # 
                # print("随机游走的最终迭代次数：",t)
                break
        return v_head   # 更新后的特征

'''定义图卷积类'''
class GCNConv(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GCNConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, adj, features):
       out = torch.mm(adj, features)  # A*X，矩阵乘
       out = torch.mm(out,self.weight)    # A*X*W
       return out


# 图卷积类
class GraphConvolution(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     stdv = 1. / np.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
       out =  adj @ features @ self.weight
    #    out = torch.mm(adj, features)  # A*X
    #    out = torch.mm(out,self.weight)    # A*X*W
       return out

###########################GCN的设计#########################
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, out_size)

    def forward(self, adj, features):   # A,X
        out = F.relu(self.gcn1(adj, features)) 
        # print("out1: ",out.shape)
        out = self.gcn2(adj, out)           
        # print("out2: ",out.shape)

        return out
    

class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin11 = FCNet(36*64+2048*2, mid_features, activate='relu')
        self.lin12 = FCNet(in_features[1], mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')
        self.lin3 = FCNet(mid_features, out_features, drop=drop)    # out_features=3129

    def forward(self, v, q):
        #x = self.fusion(self.lin11(v), self.lin12(q))
        x = self.lin11(v) * self.lin12(q)   # [batch, 1024] * [batch, 1024]
        x = self.lin2(x)    # [batch, 1024]
        x = self.lin3(x)    # [batch, 3129]
        return x

class Attention_Horizontal(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention_Horizontal, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu')  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu')   # mid_features = 1024
        self.lin = FCNet(mid_features, glimpses, drop=drop)

    def forward(self, v, q):
        """
        v = batch, num_obj=36, dim=2048
        q = batch, dim=1024
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)# 扩展维度

        x = v * q
        x = self.lin(x) # [batch, num_obj, glimps=2]
        x = F.softmax(x, dim=1)
        return x

class Attention_Vertical(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention_Vertical, self).__init__()
        self.lin_v = FCNet(36, 64, activate='relu')  # let self.lin take care of bias
        self.lin_q = FCNet(1024, 64, activate='relu')
        self.lin = FCNet(64, 64, drop=drop)     # mid_features = 1024

    def forward(self, v, q):
        """
        v = batch, num_obj=36, v_dim=2048
        q = batch, q_dim=1024
        """
        v = v.transpose(1,2)    # [batch, v_dim=2048, num_obj=36]
        v = self.lin_v(v)       # [batch, 2048, 64]
        q = self.lin_q(q)       # [batch, 64]
        batch, v_dim, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, v_dim, q_dim)# 扩展维度 [batch, 2048, 64]

        x = v * q       # [batch, 2048, 64]
        x = self.lin(x) # [batch, 2048, 64]
        x = F.softmax(x, dim=1)
        return x

def apply_attention(input, attention):
    """ 
    input = batch, dim, num_obj
    attention = batch, num_obj, glimps
    """
    batch, dim, _ = input.shape     # batch, dim=2048, _=36
    _, _, glimps = attention.shape  # _=batch, _=36, glimps=2
    x = input @ attention # batch, dim=2048, glimps=2  @:矩阵乘法
    assert(x.shape[1] == dim)
    assert(x.shape[2] == glimps)
    return x.view(batch, -1)    # [batch, dim*glimps=2048*2]
