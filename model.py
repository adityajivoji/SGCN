import torch.nn as nn
import torch.nn.functional as F
import torch

class AsymmetricConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        
        self.x1 = nn.ZeroPad2d((1, 0, 0, 0))
        self.conv1 = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=(3, 1), bias=False)
        
        self.x2 = nn.ZeroPad2d((0, 0, 1, 0))
        self.conv2 = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=(1, 3))
        
        self.shortcut = nn.Identity()

        if in_cha != out_cha:
            self.shortcut = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=1, bias=False)
    
        self.activation = nn.ReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)
        
        x1 = self.x1(x)
        x1 = self.conv1(x1)
        
        x2 = self.x2(x)
        x2 = self.conv2(x2)
        
        x2 = self.activation(x2 + x1)    

        return x2 + shortcut



class InteractionMask(nn.Module):
    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.Sequential()
        self.temporal_asymmetric_convolutions = nn.Sequential()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.add_module(
                "asymmetric_conv_" + str(i),
                AsymmetricConvolution(spatial_channels, spatial_channels),
            )
            self.temporal_asymmetric_convolutions.add_module(
                "asymmetric_conv_" + str(i),
                AsymmetricConvolution(temporal_channels, temporal_channels),
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):
        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        dense_spatial_interaction = self.spatial_asymmetric_convolutions(dense_spatial_interaction)
        dense_temporal_interaction = self.temporal_asymmetric_convolutions(dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask)
        temporal_zero = torch.zeros_like(temporal_interaction_mask)

        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                               temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):
    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, 2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x



class SelfAttention(nn.Module):

    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.tensor([d_model], dtype=torch.float64))
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

    def split_heads(self, x):
        x = x.view(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x, mask=False, multi_head=False):

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  
        else:
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, seq_len, seq_len)

        attention = attention.to(torch.float64)
        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)

        return attention, embeddings


    def forward(self, x, mask=False, multi_head=False):
        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.transpose(-1, -2))  
        else:
            attention = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, seq_len, seq_len)

        attention = attention.to(torch.float64)
        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):
    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, kernel_size=1),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()

class SparseWeightedAdjacency(torch.nn.Module):
    
    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0, 
                 number_asymmetric_conv_layer=7):
        super(SparseWeightedAdjacency, self).__init__()
        
        # dense interaction
        self.spatial_attention = SelfAttention(in_dims=spa_in_dims, d_model=embedding_dims)
        self.temporal_attention = SelfAttention(in_dims=tem_in_dims, d_model=embedding_dims)
        
        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)
        
        # interaction mask
        self.interaction_mask = InteractionMask(number_asymmetric_conv_layer=number_asymmetric_conv_layer)
        
        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()
        
    def forward(self, graph, identity):
        assert len(graph.shape) == 3
        
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)
        
        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)
        
        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)
        
        # attention fusion
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        ts_interaction = dense_temporal_interaction
        
        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)
        
        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)
        
        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)
        
        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings


import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.ReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):

        gcn_features = self.embedding(torch.matmul(adjacency, graph))

        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]




class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn_1 = GraphConvolution(in_dims, embedding_dims)
        self.spatial_temporal_sparse_gcn_2 = GraphConvolution(embedding_dims, embedding_dims)

        self.temporal_spatial_sparse_gcn_1 = GraphConvolution(in_dims, embedding_dims)
        self.temporal_spatial_sparse_gcn_2 = GraphConvolution(embedding_dims, embedding_dims)

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)
        
        gcn_spatial_features = self.spatial_temporal_sparse_gcn_1(spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)

        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn_2(gcn_spatial_features, normalized_temporal_adjacency_matrix)
        
        gcn_temporal_features = self.temporal_spatial_sparse_gcn_1(tem_graph, normalized_temporal_adjacency_matrix)
        gcn_temporal_features = gcn_temporal_features.permute(2, 1, 0, 3)
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn_2(gcn_temporal_features, normalized_spatial_adjacency_matrix)

        return gcn_spatial_temporal_features, gcn_temporal_spatial_features.permute(2, 1, 0, 3)

class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(pred_len, pred_len, kernel_size=3, padding=1),
            nn.ReLU()
        ))

        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, kernel_size=3, padding=1),
                nn.ReLU()
            ))

        self.outputs = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity):

        # graph 1 obs_len N 3

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = self.sparse_weighted_adjacency_matrices(torch.squeeze(graph), identity)

        gcn_temporal_spatial_features, gcn_spatial_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )

        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features

        gcn_representation = torch.transpose(gcn_representation, dim0=0, dim1=2)

        features = self.tcns[0](gcn_representation)

        for k in range(1, self.n_tcn):
            features = nn.Dropout(p=self.dropout)(self.tcns[k](features) + features)

        prediction = torch.mean(self.outputs(features), dim=-2)

        return torch.transpose(prediction, dim0=1, dim1=0, dim2=2)
