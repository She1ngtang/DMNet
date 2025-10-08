import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

class DiffPoolMultiScaleExtractor(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_scales=3, pool_ratios=[0.25, 0.25]):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.pool_ratios = pool_ratios

        self.scale_nodes = [num_nodes]
        for ratio in pool_ratios[:num_scales - 1]:
            self.scale_nodes.append(max(1, int(self.scale_nodes[-1] * ratio)))

        self.diffpool_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()

        prev_nodes = num_nodes
        for i in range(num_scales - 1):
            target_nodes = self.scale_nodes[i + 1]
            diffpool_layer = DiffPoolLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_nodes_in=prev_nodes,
                num_nodes_out=target_nodes
            )
            self.diffpool_layers.append(diffpool_layer)
            gcn_layer = GraphConvLayer(hidden_dim, hidden_dim)
            self.gcn_layers.append(gcn_layer)
            prev_nodes = target_nodes

        self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        # 预计算基础邻接矩阵和归一化信息
        self._precompute_adjacency_info()

        self.upsample_layers = nn.ModuleList()
        for i in range(1, num_scales):
            upsample_layer = GraphUpsampleLayer(
                hidden_dim,
                self.scale_nodes[i],
                num_nodes
            )
            self.upsample_layers.append(upsample_layer)

    def _precompute_adjacency_info(self):
        """预计算邻接矩阵相关信息以加速训练"""
        base_adj = self.build_adjacency_matrix(self.num_nodes, torch.device('cpu'))

        # 预计算度矩阵的逆平方根，避免每次forward时重复计算
        degree = torch.sum(base_adj, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_matrix = torch.diag(degree_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(degree_matrix, base_adj), degree_matrix)

        self.register_buffer('base_adj', base_adj)
        self.register_buffer('base_norm_adj', norm_adj)
        self.register_buffer('degree_inv_sqrt', degree_inv_sqrt)

    def forward(self, x):
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, num_nodes, hidden_dim)

        # 将预计算的邻接矩阵移到正确设备
        if self.base_norm_adj.device != x.device:
            self.base_adj = self.base_adj.to(x.device)
            self.base_norm_adj = self.base_norm_adj.to(x.device)

        multi_scale_features = []
        current_x = x_reshaped
        current_adj = self.base_adj
        current_norm_adj = self.base_norm_adj

        # 第一个尺度使用预计算的归一化邻接矩阵
        scale_0_features = self.gcn_layers[0](current_x, current_norm_adj)
        multi_scale_features.append((scale_0_features, current_adj))

        for i, (diffpool_layer, gcn_layer) in enumerate(zip(self.diffpool_layers, self.gcn_layers[1:])):
            # 使用优化的池化层
            pooled_x, pooled_adj, pooled_norm_adj = diffpool_layer(current_x, current_norm_adj)
            scale_features = gcn_layer(pooled_x, pooled_norm_adj)
            multi_scale_features.append((scale_features, pooled_adj))

            current_x = pooled_x
            current_adj = pooled_adj
            current_norm_adj = pooled_norm_adj

        return multi_scale_features

    def build_adjacency_matrix(self, num_nodes, device):
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        k = min(3, num_nodes - 1)  # 减少邻接数量以加速

        # 使用向量化操作构建邻接矩阵
        indices = torch.arange(num_nodes, device=device)

        for offset in range(1, k // 2 + 1):
            # 前向连接
            mask_forward = indices + offset < num_nodes
            valid_i = indices[mask_forward]
            valid_j = indices[mask_forward] + offset
            adj[valid_i, valid_j] = 1.0

            # 后向连接
            mask_backward = indices - offset >= 0
            valid_i = indices[mask_backward]
            valid_j = indices[mask_backward] - offset
            adj[valid_i, valid_j] = 1.0

        # 添加自环
        adj.fill_diagonal_(1.0)
        return adj


class DiffPoolLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes_in: int, num_nodes_out: int):
        super().__init__()
        self.num_nodes_out = num_nodes_out
        self.num_nodes_in = num_nodes_in

        # 使用更好的初始化策略减少训练时间
        self.assignment = nn.Parameter(torch.empty(num_nodes_in, num_nodes_out))
        nn.init.xavier_uniform_(self.assignment, gain=0.1)  # 更小的初始化值

        self.embed_conv = GraphConvLayer(input_dim, hidden_dim)

        # 预计算softmax的温度参数，可以调节软分配的"硬度"
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, norm_adj):
        # 使用温度参数的softmax，可以让训练更稳定更快
        assign_matrix = F.softmax(self.assignment / self.temperature.clamp(min=0.01), dim=-1)

        # 嵌入特征
        embed_x = self.embed_conv(x, norm_adj)

        # 优化的池化操作
        pooled_x = torch.matmul(assign_matrix.T, embed_x)

        # 优化池化邻接矩阵计算
        # 使用更高效的矩阵乘法顺序
        temp_adj = torch.matmul(norm_adj, assign_matrix)
        pooled_adj = torch.matmul(assign_matrix.T, temp_adj)

        # 快速归一化池化后的邻接矩阵
        pooled_norm_adj = self._fast_normalize_adj(pooled_adj)

        return pooled_x, pooled_adj, pooled_norm_adj

    def _fast_normalize_adj(self, adj):
        """快速归一化邻接矩阵"""
        # 使用更简单的行归一化而不是对称归一化
        row_sum = torch.sum(adj, dim=1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-6)
        return adj / row_sum


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)  # 使用inplace操作节省内存

        # 使用更轻量的归一化或移除归一化
        # self.norm = nn.BatchNorm1d(output_dim)  # 比LayerNorm更快
        # 或者完全移除归一化以加速训练

    def forward(self, x, norm_adj):
        # 直接使用传入的归一化邻接矩阵，避免重复计算
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        output = self.activation(output)

        # 如果使用BatchNorm，需要reshape
        # batch_size, num_nodes, hidden_dim = output.shape
        # output = output.view(-1, hidden_dim)
        # output = self.norm(output)
        # output = output.view(batch_size, num_nodes, hidden_dim)

        return output


class GraphUpsampleLayer(nn.Module):
    def __init__(self, hidden_dim, num_nodes_coarse, num_nodes_fine):
        super().__init__()
        self.num_nodes_coarse = num_nodes_coarse
        self.num_nodes_fine = num_nodes_fine
        self.upsample_proj = nn.Linear(hidden_dim, hidden_dim)
        self.refine_gcn = GraphConvLayer(hidden_dim, hidden_dim)

        # 使用更好的初始化策略
        self.upsample_weights = nn.Parameter(torch.empty(num_nodes_coarse, num_nodes_fine))
        nn.init.xavier_uniform_(self.upsample_weights, gain=0.1)

        # 添加dropout以防止过拟合并可能加速收敛
        self.dropout = nn.Dropout(0.1)

    def forward(self, coarse_features, fine_adj):
        """
        coarse_features: (B', N_coarse, H)
        fine_adj:        (N_fine, N_fine)
        return:          (B', N_fine, H)
        """
        Bp, N_coarse, H = coarse_features.shape
        N_coarse_w, N_fine = self.upsample_weights.shape
        assert N_coarse == N_coarse_w, f"上采样权重与输入节点数不一致: {N_coarse} vs {N_coarse_w}"

        # 使用softmax软分配
        S = F.softmax(self.upsample_weights, dim=0)

        # 优化的einsum操作
        up = torch.einsum('cf,bch->bfh', S, coarse_features)
        up = self.upsample_proj(up)
        up = self.dropout(up)

        # 如果fine_adj未归一化，进行快速归一化
        if not hasattr(self, '_fine_norm_adj_cache'):
            degree = torch.sum(fine_adj, dim=1, keepdim=True)
            degree = torch.clamp(degree, min=1e-6)
            fine_norm_adj = fine_adj / degree
            self._fine_norm_adj_cache = fine_norm_adj
        else:
            fine_norm_adj = self._fine_norm_adj_cache

        refined = self.refine_gcn(up, fine_norm_adj)
        return refined


class Gmamba(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, use_residual=True):
        super(Gmamba, self).__init__()
        self.num_nodes = num_nodes

        # 初始化 AdaptiveSpectralGCN 部分
        self.adaptive_adj = nn.Parameter(0.01 * torch.randn(num_nodes, num_nodes))
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # 初始化 SpatialMamba 部分
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.mamba_norm = nn.LayerNorm(input_dim)

        # 只有当维度相同时才加残差
        self.use_residual = use_residual and (input_dim == hidden_dim)

    def forward(self, x):
        """
        输入：
            x - 输入特征，形状为 (batch_size, num_nodes, seq_len, input_dim)
        输出：
            out - 经过Gmamba处理后的输出
        """
        B, N, T, H = x.shape
        assert N == self.num_nodes and H == self.linear.in_features

        # 1. 先进行图卷积：AdaptiveSpectralGCN
        # 构造对称的归一化邻接矩阵 A_hat
        A = torch.sigmoid(self.adaptive_adj)      # [N,N]，值域(0,1)
        A = (A + A.T) * 0.5                       # 对称化
        A = A + torch.eye(N, device=x.device)     # 加自环
        deg = A.sum(1)                            # [N]
        D_inv = torch.diag(1.0 / (deg + 1e-6))     # D^{-1}
        A_hat = D_inv @ A                         # 随机游走归一化

        # 图卷积
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, H)
        agg = A_hat @ x_flat

        # 线性变换并恢复维度
        gcn_out = self.linear(agg)                    # [B*T, N, hidden_dim]
        gcn_out = gcn_out.view(B, T, N, self.linear.out_features).permute(0, 2, 1, 3)  # [B, N, T, hidden_dim]
        gcn_out = self.norm(gcn_out)

        # 2. 然后进行 Mamba 操作（时空特征提取）
        xt = gcn_out.view(B * T, N, self.linear.out_features)                      # [BT, N, hidden_dim]
        gate = torch.sigmoid(self.gate(xt))           # [BT, N, hidden_dim]

        # h_{n} = tanh(h_{n-1}A + x_n B^T)  (线性递推)
        xB = torch.matmul(xt, self.B.T)               # [BT, N, hidden_dim]
        h = torch.tanh(torch.cumsum(xB, dim=1))       # 累积和近似递推

        h = h * gate
        y = torch.matmul(h, self.C.T) + torch.matmul(xt, self.D)  # [BT, N, H]
        y = self.mamba_norm(y)
        mamba_out = y.view(B, T, N, self.linear.out_features).transpose(1, 2)  # [B, N, T, hidden_dim]

        # 3. 如果需要残差连接，则加上输入
        if self.use_residual:
            mamba_out = mamba_out + x

        return mamba_out




class CrossScaleInteraction(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value, scale_mask=None):
        B, N, T, H = query.shape    # [32,170,12,64]

        # Multi-head attention
        Q = self.q_proj(query).view(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)
        K = self.k_proj(key).view(B, -1, T, self.num_heads, self.head_dim).transpose(2, 3)
        V = self.v_proj(value).view(B, -1, T, self.num_heads, self.head_dim).transpose(2, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if scale_mask is not None:
            scores = scores.masked_fill(scale_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(2, 3).contiguous().view(B, N, T, H)

        output = self.out_proj(attn_output)
        return self.norm(output + query)


class OutputPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, output_dim)
        )

    def forward(self, x):
        output = self.predictor(x)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        else:
            output = output.mean(dim=-1)
        return output


class GlobalPositionEncoding(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.node_position_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.time_position_embedding = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        B, N, T = x.shape
        assert T == self.input_dim
        x_flat = x.reshape(B * N, T)
        proj = self.input_projection(x_flat)
        proj = proj.view(B, N, self.hidden_dim)
        proj_expanded = proj.unsqueeze(2).expand(B, N, T, self.hidden_dim)
        node_pos = self.node_position_embedding.unsqueeze(0).unsqueeze(2)
        node_pos = node_pos.expand(B, N, T, self.hidden_dim)
        time_pos = self.time_position_embedding.unsqueeze(0).unsqueeze(1)
        time_pos = time_pos.expand(B, N, T, self.hidden_dim)
        fused = torch.cat([proj_expanded, node_pos, time_pos], dim=-1)
        out = self.position_fusion(fused)
        return out


class MetaGraphLearner(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_scales=3, pool_ratios=[0.25, 0.25], input_dim=170, output_dim=128):
        super(MetaGraphLearner, self).__init__()

        # 初始化GlobalPositionEncoding部分
        self.global_position_encoding = GlobalPositionEncoding(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=output_dim
        )

        # 初始化DiffPoolMultiScaleExtractor部分
        self.diffpool_extractor = DiffPoolMultiScaleExtractor(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_scales=num_scales,
            pool_ratios=pool_ratios
        )

    def forward(self, x):
        """
        输入：
            x - 输入特征，形状为 (batch_size, num_nodes, seq_len, hidden_dim)
        输出：
            out - 经过Meta Graph Learner处理后的输出
        """

        # 先进行位置编码
        position_encoded_features = self.global_position_encoding(x)

        # 通过DiffPoolMultiScaleExtractor提取多尺度特征
        multi_scale_features = self.diffpool_extractor(position_encoded_features)

        # 返回处理后的特征，结合多尺度特征和位置编码
        return multi_scale_features


class MultiScaleSpatialEncoder(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim=64, num_scales=3, dropout=0.1, pool_ratios=[0.25, 0.25]):
        super(MultiScaleSpatialEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        self.meta_graph_learner = MetaGraphLearner(hidden_dim, num_nodes, num_scales, pool_ratios, input_dim, hidden_dim)

        self.spectral_gcn_modules = nn.ModuleList()
        self.spatial_mamba_modules = nn.ModuleList()
        self.gmamba_modules = nn.ModuleList()

        for i in range(num_scales):
            scale_nodes = self.multi_scale_extractor.scale_nodes[i]
            self.gmamba_modules.append(
                Gmamba(hidden_dim, hidden_dim, scale_nodes)
            )

        self.cross_scale_interaction = CrossScaleInteraction(hidden_dim)
        self.output_predictor = OutputPredictor(hidden_dim, input_dim, dropout)
        self.residual_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, num_nodes, seq_len = x.shape
        residual = x
        multi_scale_data = self.meta_graph_learner(x)
        processed_features = []
        original_adj = self.multi_scale_extractor.build_adjacency_matrix(num_nodes, x.device)

        for i in range(self.num_scales):
            scale_features, scale_adj = multi_scale_data[i]
            scale_nodes = self.multi_scale_extractor.scale_nodes[i]
            scale_feature_4d = scale_features.view(batch_size, scale_nodes, seq_len, self.hidden_dim)
            gmamba_output = self.spatial_mamba_modules[i](scale_feature_4d)
            fused_output = gmamba_output.contiguous()

            if i == 0:
                processed_features.append(fused_output)
            else:
                fused_output_reshaped = fused_output.view(batch_size * seq_len, scale_nodes, self.hidden_dim)
                upsampled_output = self.multi_scale_extractor.upsample_layers[i - 1](fused_output_reshaped,
                                                                                     original_adj)
                upsampled_output_4d = upsampled_output.view(batch_size, num_nodes, seq_len, self.hidden_dim)
                processed_features.append(upsampled_output_4d)
        interacted_features = self.cross_scale_interaction(processed_features[0], processed_features[1], processed_features[2])
        output = self.output_predictor(interacted_features)
        output = output + self.residual_proj(residual)
        return output


