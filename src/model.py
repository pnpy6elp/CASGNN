# -*- coding: utf-8 -*-
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.sparse as dglsp
import igraph as ig
from dgl import knn_graph, to_bidirected, add_self_loop
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm 
from torch_geometric.utils import get_laplacian, from_dgl, to_undirected
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing

from src.utils import CSiLU

warnings.filterwarnings("ignore")



from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear

class CommunityGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers=1, bias=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.num_layers = num_layers

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        last_c = in_channels
        for _ in range(num_layers):
            self.linears.append(Linear(last_c, out_channels).to(torch.cfloat))
            self.norms.append(nn.LayerNorm(out_channels))
            last_c = out_channels

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        self.act = CSiLU()  

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for lin, norm in zip(self.linears, self.norms):
            h_in = h
            h = lin(h)
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            if self.bias is not None:
                h = h + self.bias
            # residual
            if h_in.shape == h.shape:
                h = h + h_in
            h = self.act(h)
        return h

import networkx as nx
from networkx.algorithms.community import louvain_communities

class IntraInterCommunityBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, combine='sum'):
        super().__init__()
        self.local_gnn = CommunityGNN(in_channels, hidden_channels)
        self.global_gnn = CommunityGNN(in_channels, hidden_channels)
        self.combine = combine  # 'sum' or 'cat'
        if combine == 'cat':
            self.out_proj = Linear(2 * hidden_channels, hidden_channels).to(torch.cfloat)
            self.out_act = CSiLU()  
            self.out_norm = nn.LayerNorm(hidden_channels)  # optional
    @torch.no_grad()
    def detect_communities(self, edge_index, num_nodes: int):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        comms = louvain_communities(G)  # list[ set(node_idx) ]
        comm_id = torch.empty(num_nodes, dtype=torch.long)
        for cid, comm in enumerate(comms):
            for n in comm:
                comm_id[n] = cid
        return comm_id

    @torch.no_grad()
    def split_edges_by_community(self, edge_index, edge_weight, comm_id):
        src, dst = edge_index  # [E], [E]
        same = (comm_id[src] == comm_id[dst])  # intra-community mask

        edge_index_intra = edge_index[:, same]
        edge_weight_intra = edge_weight[same] if edge_weight is not None else None

        edge_index_inter = edge_index[:, ~same]
        edge_weight_inter = edge_weight[~same] if edge_weight is not None else None

        return (edge_index_intra, edge_weight_intra,
                edge_index_inter, edge_weight_inter)

    def forward(
        self,
        x,
        lap=None,
        precomputed_edge_index=None,
        precomputed_edge_weight=None,
        precomputed_comm_id=None,
    ):
        if precomputed_edge_index is not None and precomputed_comm_id is not None:
            edge_index = precomputed_edge_index      # [2, E]
            edge_weight = precomputed_edge_weight    # [E] 
            comm_id = precomputed_comm_id            # [num_nodes]
        else:
            assert lap is not None, "lap or precomputed_edge_index+comm_id"
            edge_index = lap.indices()
            edge_weight = lap.val
            num_nodes = x.size(0)
            with torch.no_grad():
                comm_id = self.detect_communities(edge_index, num_nodes).to(x.device)

        (edge_index_intra, edge_weight_intra,
         edge_index_inter, edge_weight_inter) = self.split_edges_by_community(
            edge_index, edge_weight, comm_id
        )

        # 3) Local (intra-community) GNN
        h_local = self.local_gnn(x, edge_index_intra, edge_weight_intra)

        # 4) Global (inter-community) GNN
        if edge_index_inter.numel() == 0:
            h_global = torch.zeros_like(h_local)
        else:
            h_global = self.global_gnn(x, edge_index_inter, edge_weight_inter)

        if self.combine == 'sum':
            h = h_local + h_global
        else:
            h = torch.cat([h_local, h_global], dim=-1)
            h = self.out_proj(h)
            h = self.out_act(h)

        return h




class CommunityTimeModel(nn.Module):
    def __init__(
        self,
        seq_length,
        signal_length,
        pred_length,
        hidden_size,
        embed_size,
        num_ts,
        device,
        num_topk=2,   
        cd_algo='louvain',
        low_freq_ratio=0.5,
    ):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.seq_len = seq_length
        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.J = np.log(2 / np.pi) / np.log(s) + lev - 1
        self.num_ts = num_ts
        self.signal_len = signal_length
        self.cd_algo = cd_algo

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        

        self.freq_len = signal_length // 2 + 1
        self.num_nodes = self.freq_len * self.num_ts  # = N*C
        
        self.freq_len_full = signal_length // 2 + 1

        self.low_freq_ratio = low_freq_ratio
        self.freq_len = max(1, int(self.freq_len_full * self.low_freq_ratio))

        self.num_nodes = self.freq_len * self.num_ts

        lp_prior = torch.linspace(1.0, 0.5, steps=self.freq_len)

        self.register_buffer("lp_prior", lp_prior.view(1, 1, -1))  # [1,1,C]

        self.freq_gate = nn.Parameter(torch.tensor(0.0))

        self.comm_block = IntraInterCommunityBlock(
            in_channels=1,
            hidden_channels=hidden_size,
            combine='sum',
        )

        self.clin = nn.Linear(
            in_features=hidden_size * self.freq_len,
            out_features=self.freq_len
        ).to(torch.cfloat)

        self.lin2 = nn.Linear(in_features=seq_length,
                              out_features=hidden_size)
        self.lin3 = nn.Linear(in_features=hidden_size,
                              out_features=pred_length)
        self.lin4 = nn.Linear(in_features=pred_length * 2,
                              out_features=pred_length)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_length)

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_length)
        self.seasonal_gate = nn.Parameter(torch.tensor(0.0))  # scalar gate


        self.isn = nn.InstanceNorm2d(self.num_ts)

        self.act_imag = CSiLU()
        self.act_real = nn.SiLU()
        
        self.register_buffer("comm_id_single", torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_index_single", torch.empty(2, 0, dtype=torch.long))
        self.register_buffer("edge_weight_single", torch.empty(0, dtype=torch.float32))


    @torch.no_grad()
    def construct_laplacian(self, x, k, num_nodes):
        # x: [num_nodes, feat_dim]
        L = from_dgl(knn_graph(x, k, algorithm='bruteforce-sharemem'))
        L = to_undirected(
            L.edge_index.flip(0),
            torch.ones(len(L.edge_index[0]), device=self.device),
            num_nodes=num_nodes,
            reduce='min'
        )
        L = get_laplacian(L[0], L[1], normalization='sym',
                          num_nodes=num_nodes)
        return dglsp.spmatrix(L[0], L[1])

    
    @torch.no_grad()
    def detect_communities(self, edge_index, num_nodes: int):
        print(self.cd_algo)
        if self.cd_algo == 'louvain':
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            edges = edge_index.t().tolist()
            G.add_edges_from(edges)

            comms = louvain_communities(G)
            comm_id = torch.empty(num_nodes, dtype=torch.long)
            for cid, comm in enumerate(comms):
                for n in comm:
                    comm_id[n] = cid
        elif self.cd_algo == 'leiden':
            edges = edge_index.t().tolist()
            g = ig.Graph(n=num_nodes, edges=edges, directed=False)

            partition = g.community_leiden(objective_function='modularity')
            comm_id = partition.membership
            comm_id = torch.tensor(comm_id)
        elif self.cd_algo == 'label_propagation':
            edges = edge_index.t().tolist()
            g = ig.Graph(n=num_nodes, edges=edges, directed=False)
            partition = g.community_label_propagation()
            comm_id = partition.membership
            comm_id = torch.tensor(comm_id)
        elif self.cd_algo == "infomap":
            edges = edge_index.t().tolist()
            g = ig.Graph(n=num_nodes, edges=edges, directed=False)
            
            partition = g.community_infomap()
            comm_id = partition.membership
            comm_id = torch.tensor(comm_id)
        elif self.cd_algo == "fastgreedy":
            edges = edge_index.t().tolist()
            g = ig.Graph(n=num_nodes, edges=edges, directed=False)
            g.simplify(multiple=True, loops=True)

            dendrogram = g.community_fastgreedy()
            partition = dendrogram.as_clustering()
            comm_id = torch.tensor(partition.membership)
            
        print(f"The # of Communities: {len(set(comm_id.tolist()))}")
        print(f"The # of Nodes: {num_nodes}")
            
            
            
        return comm_id

    @torch.no_grad()
    def init_single_graph(self, x_fft_first_sample):
        num_nodes = self.num_nodes  # = N*C

        # [N, C] -> [num_nodes]
        x0 = x_fft_first_sample.reshape(num_nodes)

        feat0 = torch.stack([x0.real, x0.imag], dim=-1)  # [num_nodes, 2]

        lap = self.construct_laplacian(feat0, self.k, num_nodes=num_nodes)
        edge_index = lap.indices()
        edge_weight = lap.val

        comm_id = self.detect_communities(edge_index.cpu(), num_nodes)
        self.comm_id_single = comm_id.to(self.device)
        self.edge_index_single = edge_index
        self.edge_weight_single = edge_weight


    @torch.no_grad()
    def init_communities(self, lap):
        edge_index = lap.indices()
        edge_weight = lap.val
        num_nodes = lap.shape[0]

        comm_id = self.detect_communities(edge_index.cpu(), num_nodes)
        self.comm_id = comm_id.to(self.device)
        self.edge_index_full = edge_index
        self.edge_weight_full = edge_weight

    def forward(self, x):
        # x: [B, seq_len, num_ts]
        seasonal_init, trend_init = self.decompsition(x)   # [B, L, N]

        trend_init    = trend_init.permute(0, 2, 1)        # [B, N, L]
        seasonal_time = seasonal_init.permute(0, 2, 1)     # [B, N, L]

        trend_output = self.Linear_Trend(trend_init)       # [B, N, pred_len]

        x_fft = torch.fft.rfft(seasonal_init, n=self.signal_len, dim=1, norm='ortho')
        B, C_full, N = x_fft.shape  # [B, C_full, N]

        C = self.freq_len
        x_fft = x_fft[:, :C, :]     # [B, C, N]

        assert self.num_nodes == C * N, "num_nodes 설정이 잘못되었습니다."

        # [B, C, N] -> [B, N, C] 
        x_fft = x_fft.permute(0, 2, 1).contiguous()   # [B, N, C]
        nodes_per_graph = self.num_nodes              # = N*C
        batch_total_nodes = B * nodes_per_graph

        if self.comm_id_single.numel() == 0:
            self.init_single_graph(x_fft[0])

        x_nodes = x_fft.reshape(B, nodes_per_graph)           # [B, num_nodes]
        x_nodes = x_nodes.reshape(batch_total_nodes, 1)       # [B*num_nodes, 1]

        # 3) batched edge index
        base_ei = self.edge_index_single                      # [2, E]
        E = base_ei.size(1)
        offsets = torch.arange(B, device=x.device).view(B, 1, 1) * nodes_per_graph
        ei_batched = base_ei.view(1, 2, E) + offsets
        ei_batched = ei_batched.view(2, B * E)
        ew_batched = self.edge_weight_single.repeat(B)
        comm_id_batched = self.comm_id_single.repeat(B)

        # 4) intra / inter community GNN
        x_nodes = self.comm_block(
            x_nodes,
            lap=None,
            precomputed_edge_index=ei_batched,
            precomputed_edge_weight=ew_batched,
            precomputed_comm_id=comm_id_batched,
        )

        # 5) [B, N, C, H]
        x_nodes = x_nodes.reshape(B, N, C, -1)                # [B, N, C, H]
        x_nodes = x_nodes.permute(0, 1, 3, 2).reshape(B, N, -1)

        x_spec = self.clin(x_nodes)  # [B, N, C], complex

        gate = torch.tanh(self.freq_gate)  # scalar in (-1,1)

        lp = self.lp_prior  # [1,1,C]

        x_spec = x_spec * (1.0 + gate * (lp - 1.0))
        x_time = torch.fft.irfft(x_spec, n=self.seq_len, dim=-1, norm='ortho')  # [B, N, seq_len]

        x_time = self.isn(x_time)
        x_time = self.act_real(x_time)
        x_time = self.lin2(x_time)
        x_time = self.isn(x_time)
        x_time = self.act_real(x_time)
        x_time = self.lin3(x_time)                    # [B, N, pred_len]

        seasonal_output = self.Linear_Seasonal(seasonal_time)  # [B, N, pred_len]

        g = torch.sigmoid(self.seasonal_gate)  # scalar in (0,1)

        x_time = x_time + g * seasonal_output  # [B, N, pred_len]

        x_out = torch.cat((x_time, trend_output), dim=-1)  # [B, N, pred_len*2]
        x_out = self.lin4(x_out)

        return x_out






class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

