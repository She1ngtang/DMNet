import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.FloatTensor([dim]))

    def forward(self, x):
        device = x.device
        self.scale = self.scale.to(device)
        # x: (batch_size, seq_length, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_probs, V)

        return out


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.FloatTensor([dim]))

    def forward(self, x, y):
        device = x.device
        device = y.device
        self.scale = self.scale.to(device)
        # x, y: (batch_size, seq_length, dim)
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_probs, V)

        return out


class SpatioTemporalIntegrator(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SpatioTemporalIntegrator, self).__init__()
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )
        self.self_attention_s = SelfAttention(dim)
        self.self_attention_t = SelfAttention(dim)
        self.cross_attention_st = CrossAttention(dim)
        self.cross_attention_ts = CrossAttention(dim)

        # Fully Connected layer for combining features
        self.fc = nn.Linear(dim, dim)

    def forward(self, S, T):
        device = S.device
        self.to(device)
        S_mapped = self.out_cross_layer(S)
        T_mapped = self.out_cross_layer(T)
        S_self_attended = self.self_attention_s(S_mapped)
        T_self_attended = self.self_attention_t(T_mapped)

        S_cross_attended = self.cross_attention_st(S_mapped, T_mapped)
        T_cross_attended = self.cross_attention_ts(T_mapped, S_mapped)
        S_combined = S_self_attended + S_cross_attended
        T_combined = T_self_attended + T_cross_attended
        combined = S_combined + T_combined
        combined_fc = self.fc(combined)
        output = combined_fc * T + S
        return output