
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

# modelname = "/fred/oz064/wenyuzhao/ExperimentalDataset/BERTVariants/scibert_scivocab_uncased/"
# config = BertConfig.from_json_file('/fred/oz064/wenyuzhao/ExperimentalDataset/BERTVariants/scibert_scivocab_uncased/bert_config.json')
modelname= "allenai/scibert_scivocab_uncased"
config = BertConfig.from_pretrained(osp.join(modelname,'config.json'), output_hidden_states=True, output_attentions=True)


class MLPModel(nn.Module):
    def __init__(self, ninp, nout, nhid):
        super(MLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.mol_hidden1 = nn.Linear(nout, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)
        

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained(modelname, config=config)
        self.text_transformer_model.train()

        self.device = 'GPU'

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, molecule, text_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        x = self.relu(self.mol_hidden1(molecule))
        x = self.relu(self.mol_hidden2(x))
        x = self.mol_hidden3(x)


        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)


        return text_x, x


class GCNModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels):
        super(GCNModel, self).__init__()
        

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #For GCN:
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)


        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained(modelname,config=config)
        self.text_transformer_model.train()

        self.device = 'GPU'

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)


        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        print(x.shape, edge_index.shape)
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x).relu()
        x = self.mol_hidden3(x)


        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x


class AttentionModel(nn.Module):

    def __init__(self, num_node_features, ninp, nout, nhid, nhead, nlayers, graph_hidden_channels, mol_trunc_length, temp, dropout=0.5):
        super(AttentionModel, self).__init__()
        
        self.text_hidden1 = nn.Linear(ninp, nhid)
        self.text_hidden2 = nn.Linear(nhid, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.num_node_features = num_node_features
        self.graph_hidden_channels = graph_hidden_channels
        self.mol_trunc_length = mol_trunc_length

        self.drop = nn.Dropout(p=dropout)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.text_transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        

        self.temp = nn.Parameter(torch.Tensor([temp]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #For GCN:
        self.conv1 = GCNConv(self.num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)


        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained(modelname,config=config)
        self.text_transformer_model.train()

        self.device = 'GPU'

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        mol_x = self.conv3(x, edge_index)

        #turn pytorch geometric output into the correct format for transformer
        #requires recovering the nodes from each graph into a separate dimension
        node_features = torch.zeros((graph_batch.num_graphs, self.mol_trunc_length, self.graph_hidden_channels)).to(self.device)
        for i, p in enumerate(graph_batch.ptr):
          if p == 0: 
            old_p = p
            continue
          node_features[i - 1, :p-old_p, :] = mol_x[old_p:torch.min(p, old_p + self.mol_trunc_length), :]
          old_p = p
        node_features = torch.transpose(node_features, 0, 1)

        text_output = self.text_transformer_decoder(text_encoder_output['last_hidden_state'].transpose(0,1), node_features, 
                                                            tgt_key_padding_mask = text_mask == 0, memory_key_padding_mask = ~molecule_mask) 


        #Readout layer
        x = global_mean_pool(mol_x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x)
        x = x.relu()
        x = self.mol_hidden2(x)

        text_x = torch.tanh(self.text_hidden1(text_output[0,:,:])) #[CLS] pooler
        text_x = self.text_hidden2(text_x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x


class cross_modal_discriminator(nn.Module):
    def __init__(self, input_dim):
        super(cross_modal_discriminator, self).__init__()
        self.input_dim =input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, 300),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(300, 1)
        )
        self.device = 'GPU'

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, fea):
        output = self.model(fea)

        return output


class GraphSageRMLPModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels):
        super(GraphSageRMLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter('temp', self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        # For GCN:
        self.conv1 = SAGEConv(num_node_features, graph_hidden_channels)
        self.conv2 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = SAGEConv(graph_hidden_channels, graph_hidden_channels)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nout)
        # self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        # self.mol_hidden2 = nn.Linear(nhid, nhid)
        # self.mol_hidden3 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters())  # get all but bert params

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        # Obtain node embeddings
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        # print("look up shape:", x.shape, edge_index.shape, batch)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x).relu()
        # x = self.mol_hidden2(x).relu()
        # x = self.mol_hidden3(x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x


class GraphAttentionRMLPModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels, heads):
        super(GraphAttentionRMLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.heads = heads

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter('temp', self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        # For GCN:
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads)
        self.conv2 = GATConv(graph_hidden_channels*heads, graph_hidden_channels*heads, heads)
        self.conv3 = GATConv(graph_hidden_channels*heads*heads, graph_hidden_channels)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nout)
        # self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        # self.mol_hidden2 = nn.Linear(nhid, nhid)
        # self.mol_hidden3 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters())  # get all but bert params

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        # Obtain node embeddings
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        # print("look up shape:", x.shape, edge_index.shape, batch)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x).relu()
        # x = self.mol_hidden2(x).relu()
        # x = self.mol_hidden3(x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x


class SuperGATRMLPModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels):
        super(SuperGATRMLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        # self.heads =heads
        # self.num_layers = num_layers


        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter('temp', self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        # For GCN:
        self.conv1 = SuperGATConv(num_node_features, graph_hidden_channels)
        self.conv2 = SuperGATConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = SuperGATConv(graph_hidden_channels, graph_hidden_channels)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nout)
        # self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        # self.mol_hidden2 = nn.Linear(nhid, nhid)
        # self.mol_hidden3 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters())  # get all but bert params

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        # Obtain node embeddings
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        # print("look up shape:", x.shape, edge_index.shape, batch)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x).relu()
        # x = self.mol_hidden2(x).relu()
        # x = self.mol_hidden3(x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x


##
class TransformerRMLPModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels, heads):
        super(TransformerRMLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.num_heads = heads


        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter('temp', self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        # For GCN:
        self.conv1 = TransformerConv(num_node_features, graph_hidden_channels, heads)
        self.conv2 = TransformerConv(graph_hidden_channels * heads, graph_hidden_channels * heads, heads)
        self.conv3 = TransformerConv(graph_hidden_channels * heads * heads, graph_hidden_channels)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nout)
        # self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        # self.mol_hidden2 = nn.Linear(nhid, nhid)
        # self.mol_hidden3 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters())  # get all but bert params

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask=None, molecule_mask=None):
        text_encoder_output = self.text_transformer_model(text, attention_mask=text_mask)


        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)


        # Obtain node embeddings
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch


        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x).relu()


        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)


        return text_x, x



