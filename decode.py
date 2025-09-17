from transformers import BertTokenizerFast, BertModel,BertConfig

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv,SAGEConv, GATConv,TransformerConv, SuperGATConv
from torch_geometric.nn import global_mean_pool, GlobalAttention,global_max_pool,global_add_pool
from sklearn.decomposition import TruncatedSVD
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from scipy import linalg

import os.path as osp

# class stepwisePivot (nn.Module):
#     def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels, heads):
#         super(stepwisePivot,self).__init__()

# 定义一个用于处理单视图的Transformer块
class SingleViewTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pivot):
        # 拼接视图序列和枢纽
        input_seq = torch.cat([x, pivot], dim=1)

        # MHA
        attn_output, _ = self.self_attn(input_seq, input_seq, input_seq)
        attn_output = self.dropout1(attn_output)
        # 残差连接和LN
        input_seq = self.norm1(input_seq + attn_output)

        # FFN
        ffn_output = self.ffn(input_seq)
        ffn_output = self.dropout2(ffn_output)
        # 残差连接和LN
        output_seq = self.norm2(input_seq + ffn_output)

        # 分割回视图序列和枢纽
        output_x = output_seq[:, :-1, :]
        output_pivot = output_seq[:, -1, :].unsqueeze(1)

        return output_x, output_pivot


# 主模型类
class StepwisePivotTransformer(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels, heads, num_layers=3):
        super().__init__()

        self.ninp = ninp
        self.nout = nout
        self.num_heads = heads
        self.num_layers = num_layers
        self.graph_hidden_channels = graph_hidden_channels

        # 文本视图处理
        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()
        self.text_projection = nn.Linear(ninp, nout)

        # 图视图处理
        self.conv1 = TransformerConv(num_node_features, graph_hidden_channels, heads)
        self.conv2 = TransformerConv(graph_hidden_channels * heads, graph_hidden_channels * heads, heads)
        self.conv3 = TransformerConv(graph_hidden_channels * heads * heads, graph_hidden_channels)
        self.mol_projection = nn.Linear(graph_hidden_channels, nout)

        # 枢纽交互层
        self.pivot_layers = nn.ModuleList([
            nn.ModuleDict({
                'text': SingleViewTransformerBlock(nout, heads, nhid),
                'graph': SingleViewTransformerBlock(nout, heads, nhid)
            }) for _ in range(num_layers)
        ])

        # 最终融合特征的MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(nout, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout)
        )

        self.ln = nn.LayerNorm((nout))

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        # 1. 初始输入编码
        # 文本视图
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)
        # 这里的text_x的形状是 (batch_size, sequence_length, ninp)
        text_x = text_encoder_output.last_hidden_state
        # 投影到nout维度
        text_x = self.text_projection(text_x)

        # 图视图
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 图的readout层
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]
        # 投影到nout维度
        graph_x = self.mol_projection(x)
        # 扩展维度以匹配text_x的序列维度，这里我们假设graph_x只有一个token
        graph_x = graph_x.unsqueeze(1)

        # 2. 初始枢纽
        # 假设枢纽的形状与视图序列相同，这里简化为一个单token的序列
        pivot_init = (torch.mean(text_x, dim=1, keepdim=True) + graph_x) / 2
        # pivot_init = (text_x[:, 0:1, :] + graph_x) / 2 # 也可以用CLS token

        pivot = pivot_init

        # 3. 逐步枢纽融合
        for i in range(self.num_layers):
            # 步骤1: 融合文本视图信息
            text_x, temp_pivot_text = self.pivot_layers[i]['text'](text_x, pivot)
            pivot = (temp_pivot_text + pivot) / 2

            # 步骤2: 融合图视图信息
            graph_x, temp_pivot_graph = self.pivot_layers[i]['graph'](graph_x, pivot)
            pivot = (temp_pivot_graph + pivot) / 2

        # 4. 最终输出
        # 全局融合特征：从最终的枢纽中提取
        f_c = self.final_mlp(pivot.squeeze(1))

        # 领域特定特征：可以从最终的文本和图序列中提取
        # 示例：取序列的均值作为领域特定特征
        f_text = torch.mean(text_x, dim=1)
        f_graph = graph_x.squeeze(1)

        # 对最终的特征进行归一化
        f_c = self.ln(f_c)
        f_text = self.ln(f_text)
        f_graph = self.ln(f_graph)

        return f_c, f_text, f_graph