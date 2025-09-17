import torch  # 导入 PyTorch 库，用于构建神经网络
from torch import nn  # 从 PyTorch 中导入 nn 模块，其中包含了所有神经网络层
import torch.nn.functional as F  # 导入 nn.functional 模块，包含了各种无状态的函数，如激活函数、损失函数等
import numpy as np  # 导入 NumPy 库，用于进行数值计算
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, \
    SuperGATConv  # 导入 PyTorch Geometric 库中的图神经网络层
from torch_geometric.nn import global_mean_pool  # 导入图神经网络中的全局池化函数
from sklearn.decomposition import TruncatedSVD  # 导入 Sklearn 的截断SVD，用于降维
from torch.nn import TransformerDecoder, TransformerDecoderLayer  # 导入 Transformer Decoder 模块
from scipy import linalg  # 导入 SciPy 的线性代数模块
from transformers import BertTokenizerFast, BertModel, BertConfig  # 导入 Hugging Face 的 transformers 库中的 BERT 模型和分词器

import os.path as osp  # 导入 os.path 模块，用于处理文件路径


# --- 主模型实现：StepwisePivotTransformer ---

# 定义一个用于处理单视图的Transformer块
class SingleViewTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()  # 调用父类 nn.Module 的构造函数
        # 定义多头注意力层，用于处理拼接后的视图和枢纽序列
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一层 LayerNorm
        # 定义前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),  # FFN 的第一层线性变换
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Dropout(dropout),  # dropout 层
            nn.Linear(dim_feedforward, d_model)  # FFN 的第二层线性变换
        )
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二层 LayerNorm
        self.dropout1 = nn.Dropout(dropout)  # 定义第一个 dropout 层
        self.dropout2 = nn.Dropout(dropout)  # 定义第二个 dropout 层

    def forward(self, x, pivot):
        # 拼接视图序列 (x) 和枢纽序列 (pivot)，以便一起输入 Transformer
        input_seq = torch.cat([x, pivot], dim=1)

        # 多头注意力（MHA）计算
        attn_output, _ = self.self_attn(input_seq, input_seq, input_seq)
        attn_output = self.dropout1(attn_output)
        # 加上残差连接和第一层 LayerNorm
        input_seq = self.norm1(input_seq + attn_output)

        # 前馈网络（FFN）计算
        ffn_output = self.ffn(input_seq)
        ffn_output = self.dropout2(ffn_output)
        # 加上残差连接和第二层 LayerNorm
        output_seq = self.norm2(input_seq + ffn_output)

        # 将处理后的序列分割回更新后的视图序列和枢纽
        output_x = output_seq[:, :-1, :]  # 视图序列是除了最后一个 token
        output_pivot = output_seq[:, -1, :].unsqueeze(1)  # 枢纽是最后一个 token pivot的sequence_length值为1

        return output_x, output_pivot


# 主模型类，实现了逐步枢纽 Transformer
class StepwisePivotTransformer(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels, heads, num_layers=3):
        super().__init__()  # 调用父类 nn.Module 的构造函数

        self.ninp = ninp  # BERT 输出的维度
        self.nout = nout  # 模型的输出维度，也作为 Transformer 层的 d_model
        self.num_heads = heads  # 多头注意力的头数
        self.num_layers = num_layers  # 逐步融合的层数
        self.graph_hidden_channels = graph_hidden_channels  # 图GNN的隐藏层维度

        # 文本视图处理：加载预训练的 SciBERT 模型
        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()  # 设置模型为训练模式
        self.text_projection = nn.Linear(ninp, nout)  # 线性层，将BERT输出的维度投影到模型维度

        # 图视图处理：定义图神经网络层
        self.conv1 = TransformerConv(num_node_features, graph_hidden_channels, heads)
        self.conv2 = TransformerConv(graph_hidden_channels * heads, graph_hidden_channels * heads, heads)
        self.conv3 = TransformerConv(graph_hidden_channels * heads * heads, graph_hidden_channels)
        self.mol_projection = nn.Linear(graph_hidden_channels, nout)  # 线性层，将图表示投影到模型维度

        # 枢纽交互层：多层 Stepwise Pivot 结构
        self.pivot_layers = nn.ModuleList([
            nn.ModuleDict({
                'text': SingleViewTransformerBlock(nout, heads, nhid),  # 文本视图的Transformer块
                'graph': SingleViewTransformerBlock(nout, heads, nhid)  # 图视图的Transformer块
            }) for _ in range(num_layers)
        ])

        # 最终融合特征的MLP，用于从枢纽中提取全局特征
        self.final_mlp = nn.Sequential(
            nn.Linear(nout, nhid),  # 第一层线性变换
            nn.ReLU(),  # 激活函数
            nn.Linear(nhid, nhid),  # 第二层线性变换
            nn.ReLU(),  # 激活函数
            nn.Linear(nhid, nout)  # 输出层
        )

        self.ln = nn.LayerNorm((nout))  # 定义 LayerNorm 层，用于最终的特征归一化

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        # 1. 初始输入编码
        # 文本视图：通过BERT获取序列输出
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)
        # 获取BERT的最后一层隐藏状态，形状为 (batch_size, sequence_length, ninp)
        text_x = text_encoder_output.last_hidden_state
        # 通过线性层投影到 nout 维度
        text_x = self.text_projection(text_x)

        # 图视图：通过GNN编码图
        x = graph_batch.x  # 节点特征
        edge_index = graph_batch.edge_index  # 边索引
        batch = graph_batch.batch  # 批次信息，用于区分不同图的节点
        x = self.conv1(x, edge_index)  # GNN层1
        x = x.relu()
        x = self.conv2(x, edge_index)  # GNN层2
        x = x.relu()
        x = self.conv3(x, edge_index)  # GNN层3
        # 使用全局平均池化（Readout Layer）得到图的表示
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]
        # 通过线性层投影到 nout 维度
        graph_x = self.mol_projection(x)
        # 扩展维度，使图表示成为一个单 token 序列，方便与文本序列拼接
        graph_x = graph_x.unsqueeze(1)

        # 2. 初始枢纽
        # 将文本序列的均值和图表示的平均值作为初始枢纽
        pivot_init = (torch.mean(text_x, dim=1, keepdim=True) + graph_x) / 2
        pivot = pivot_init

        # 3. 逐步枢纽融合
        for i in range(self.num_layers):  # 循环 num_layers 次，进行多层融合
            # 步骤1: 融合文本视图信息
            # 将文本序列和当前枢纽输入到文本视图的 Transformer 块中
            text_x, temp_pivot_text = self.pivot_layers[i]['text'](text_x, pivot)
            # 根据公式 P = (P_temp + P_orig) / 2 更新枢纽
            pivot = (temp_pivot_text + pivot) / 2

            # 步骤2: 融合图视图信息
            # 将图表示和当前枢纽输入到图视图的 Transformer 块中
            graph_x, temp_pivot_graph = self.pivot_layers[i]['graph'](graph_x, pivot)

            # 再次更新枢纽，使其包含图的信息
            pivot = (temp_pivot_graph + pivot) / 2

        # 4. 最终输出
        # 全局融合特征：通过 MLP 从最终的枢纽中提取
        f_c = self.final_mlp(pivot.squeeze(1))

        # 领域特定特征：从最终的文本和图序列中提取
        f_text = torch.mean(text_x, dim=1)  # 文本特征，取序列的均值
        f_graph = graph_x.squeeze(1)  # 图特征

        # 对最终的特征进行 LayerNorm 归一化
        f_c = self.ln(f_c)
        f_text = self.ln(f_text)
        f_graph = self.ln(f_graph)

        return f_c, f_text, f_graph  # 返回全局融合特征和两个领域的特定特征