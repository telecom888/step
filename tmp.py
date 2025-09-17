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