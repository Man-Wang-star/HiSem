import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hisem.CascadeMoE import Implement_MoE

class FeatModulator(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        hidden_dim = dim // reduction
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        )
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_norm = nn.LayerNorm(dim)

    def _enhance_2d(self, x):
        B, L, D = x.shape
        H = W = int(L ** 0.5)  # 7
        x_2d = x.transpose(1, 2).view(B, D, H, W)  # [B, D, 7, 7]
        x_2d = x_2d + self.local_conv(x_2d)
        x_2d = x_2d.view(B, D, -1).transpose(1, 2)  # [B, 49, D]
        return self.local_norm(x_2d)

    def forward(self, feat_before, feat_after):
        B, L, D = feat_before.shape
        diff = torch.abs(feat_before - feat_after)  # [B, 49, D]
        gate = self.gate_net(torch.cat([feat_before, feat_after], dim=-1))  # [B, 49, D]
        change_attn = gate * diff  # [B, 49, D]

        enhanced_b = self._enhance_2d(feat_before) + change_attn
        enhanced_a = self._enhance_2d(feat_after) + change_attn
        return enhanced_b, enhanced_a, change_attn


class BidirectionalDiffModule(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, init_lambda=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lambda_val = init_lambda

        # Cross: Before → After
        self.q_b = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_a = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_a = nn.Linear(dim, dim, bias=qkv_bias)

        # Cross: After → Before
        self.q_a = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_b = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_b = nn.Linear(dim, dim, bias=qkv_bias)

        self.fuse_gate_b = nn.Linear(dim * 2, dim)
        self.fuse_gate_a = nn.Linear(dim * 2, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)

        self.norm_add = nn.LayerNorm(dim)
        self.norm_del = nn.LayerNorm(dim)

    def forward(self, feat_before, feat_after, change_add_bias, change_del_bias):
        B, L, D = feat_before.shape
        def reshape(x):
            return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_b = reshape(self.q_b(feat_before))
        k_b = reshape(self.k_b(feat_before))
        v_b = reshape(self.v_b(feat_before))

        q_a = reshape(self.q_a(feat_after))
        k_a = reshape(self.k_a(feat_after))
        v_a = reshape(self.v_a(feat_after))

        attn_b2a_raw = (q_b @ k_a.transpose(-2, -1)) * self.scale  # [B, h, L, L]
        attn_a2b_raw = (q_a @ k_b.transpose(-2, -1)) * self.scale

        attn_b2a = F.softmax(attn_b2a_raw + change_add_bias, dim=-1)
        attn_a2b = F.softmax(attn_a2b_raw + change_del_bias, dim=-1)

        attn_add = attn_b2a - self.lambda_val * attn_a2b
        attn_del = attn_a2b - self.lambda_val * attn_b2a

        out_add = (attn_add @ v_a).transpose(1, 2).reshape(B, L, D)
        out_del = (attn_del @ v_b).transpose(1, 2).reshape(B, L, D)

        gate_b = torch.sigmoid(self.fuse_gate_b(torch.cat([feat_before, out_del], dim=-1)))
        output_b = self.proj_b(gate_b * out_del + (1 - gate_b) * feat_before)
        gate_a = torch.sigmoid(self.fuse_gate_a(torch.cat([feat_after, out_add], dim=-1)))
        output_a = self.proj_a(gate_a * out_add + (1 - gate_a) * feat_after)
        return output_b, output_a

class TripleDifferentialEnhancer(nn.Module):
    def __init__(self, dim, n_layers, feature_size, attention_module_kwargs=None, dropout=.1, num_heads=8, reduction=4, init_lambda=0.9):
        super().__init__()
        self.n_layers = n_layers
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.channels = channels

        self.feat_modulators = nn.ModuleList([])
        self.bidiff_modules = nn.ModuleList([])
        self.norm_b_list = nn.ModuleList([])
        self.norm_a_list = nn.ModuleList([])
        self.del_predictor_list = nn.ModuleList([])
        self.add_predictor_list = nn.ModuleList([])

        self.add_scale = nn.Parameter(torch.tensor(1.0))
        self.del_scale = nn.Parameter(torch.tensor(0.5))

        for i in range(n_layers):
            self.feat_modulators.append(FeatModulator(dim, reduction))
            self.bidiff_modules.append(BidirectionalDiffModule(dim, num_heads, init_lambda=init_lambda))
            self.norm_b_list.append(nn.LayerNorm(dim))
            self.norm_a_list.append(nn.LayerNorm(dim))
            self.add_predictor_list.append(nn.Linear(dim, 1))
            self.del_predictor_list.append(nn.Linear(dim, 1))

    def forward(self, feat_before, feat_after):
        b, a = feat_before, feat_after
        N, L, D = a.shape

        for l in range(self.n_layers):
            out_b, out_a, change_attn = self.feat_modulators[l](b, a)

            add_score = self.add_predictor_list[l](change_attn).squeeze(-1)
            del_score = self.del_predictor_list[l](change_attn).squeeze(-1)
            add_bias = add_score.unsqueeze(1).unsqueeze(1)
            del_bias = del_score.unsqueeze(1).unsqueeze(1)
            add_bias = add_bias * self.add_scale
            del_bias = del_bias * self.del_scale
            change_add_bias_trans = torch.diag_embed(add_bias, dim1=-2, dim2=-1).squeeze(dim=1)  # [B, 1, L, L]
            change_del_bias_trans = torch.diag_embed(del_bias, dim1=-2, dim2=-1).squeeze(dim=1)  # [B, 1, L, L]

            refined_b, refined_a = self.bidiff_modules[l](out_b, out_a, change_add_bias_trans, change_del_bias_trans)
            b, a = self.norm_b_list[l](refined_b), self.norm_a_list[l](refined_a)

        return b, a