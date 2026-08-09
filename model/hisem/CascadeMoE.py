import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Expert(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoEGate(nn.Module):
    def __init__(self, d_model, top_k, num_experts, n_group, topk_group):
        super(MoEGate, self).__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts
        self.routed_scaling_factor = 1.0
        self.scoring_func = "sigmoid"
        self.alpha = 0.001
        self.seq_aux = True
        self.topk_method = "group_limited_greedy"
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = False
        self.gating_dim = d_model
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    #【B,L,D】
        hidden_states = hidden_states.contiguous().view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores     #【B×L，n_routed_experts】
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)      # [B, L * top_k]
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)    #【B,L，n_routed_experts】
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class SparseMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, d_model, d_ff, dropout, attention_module_kwargs):
        super(SparseMoE, self).__init__()
        self.d_model = d_model
        self.top_k = attention_module_kwargs.get("top_k")
        self.num_experts = attention_module_kwargs.get("num_experts")
        self.n_shared_experts = attention_module_kwargs.get("n_shared_experts")
        self.n_group = attention_module_kwargs.get("n_group")
        self.topk_group = attention_module_kwargs.get("topk_group")
        self.ep_size = 1
        self.experts_per_rank = self.num_experts
        self.d_ff = d_ff
        self.dropout = dropout
        self.ep_rank = 0
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for i in range(self.num_experts)])
        self.gate = MoEGate(d_model, self.top_k, self.num_experts, self.n_group, self.topk_group)

        if self.n_shared_experts is not None:
            intermediate_size = d_model * self.n_shared_experts
            self.shared_experts = Expert(d_model, intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.top_k, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class CLSGate(nn.Module):
    def __init__(self, d_model, feature_size):
        super(CLSGate, self).__init__()
        self.top_k = 1
        self.n_routed_experts = 2
        self.routed_scaling_factor = 1.0
        self.scoring_func = "softmax"
        self.topk_method = "greedy"
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.channels = channels
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.gating_dim = d_model
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    #【B,L,D】
        hidden_states = hidden_states.permute(0, 2, 1).view(bsz, h, self.h_feat, self.w_feat)
        hidden_states = self.pooling(hidden_states).squeeze(-1).squeeze(-1)  # [B, D]
        ### compute gating score
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )     #【B，n_routed_experts】
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight, logits

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''
    def __init__(self, d_model=768, d_ff=2048, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out)
        return out


class resblock1(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock1, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)

class resblock2(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock2, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)

class fuse1(nn.Module):
    def __init__(self, d_model, channels, feature_size):
        super(fuse1, self).__init__()
        self.d_model = d_model
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.channels = channels
        self.Conv1_list = nn.Conv2d(channels * 2, d_model, kernel_size=1)
        self.LN_list = resblock1(d_model, d_model)
        self.LN_norm = nn.LayerNorm(channels)

    def forward(self, output_1, output_2):
        img_list = []
        batch, c = output_1.shape[0], output_1.shape[-1]
        h, w = self.h_feat, self.w_feat
        # bitemporal fusion
        output_2 = output_2.view(batch, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        output_1 = output_1.view(batch, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        feat_cap = torch.cat([output_1, output_2], dim=1)  # [B, 2C, H, W]
        feat_cap = self.LN_list(self.Conv1_list(feat_cap))
        img_fuse = feat_cap.view(batch, c, -1).transpose(-1, 1)   # (batch_size, L, D)
        img_fuse = self.LN_norm(img_fuse).unsqueeze(-1)
        img_list.append(img_fuse)
        feat_cap = img_list[-1][..., 0]  # [B,L,D]
        return feat_cap

class fuse2(nn.Module):
    def __init__(self, d_model, channels, feature_size):
        super(fuse2, self).__init__()
        self.d_model = d_model
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.channels = channels
        self.Conv1_list = nn.Conv2d(channels * 2, d_model, kernel_size=1)
        self.LN_list = resblock2(d_model, d_model)
        self.LN_norm = nn.LayerNorm(channels)

    def forward(self, output_1, output_2):
        img_list = []
        batch, c = output_1.shape[0], output_1.shape[-1]
        h, w = self.h_feat, self.w_feat
        # bitemporal fusion
        output_2 = output_2.view(batch, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        output_1 = output_1.view(batch, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        feat_cap = torch.cat([output_1, output_2], dim=1)
        feat_cap = self.LN_list(self.Conv1_list(feat_cap))
        img_fuse = feat_cap.view(batch, c, -1).transpose(-1, 1)   # (batch_size, L, D)
        img_fuse = self.LN_norm(img_fuse).unsqueeze(-1)
        img_list.append(img_fuse)
        feat_cap = img_list[-1][..., 0]  # [B,L,D]
        return feat_cap

class Implement_MoE(nn.Module):

    def __init__(self, d_model, d_ff, dropout, feature_size, attention_module_kwargs):
        super(Implement_MoE, self).__init__()
        self.d_model = d_model
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.channels = channels
        self.d_ff = d_ff
        self.dropout = dropout

        self.changemoe = SparseMoE(d_model, d_ff, dropout, attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.clsgate = CLSGate(d_model, feature_size)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.reblock1 = fuse1(d_model, channels, feature_size)
        self.reblock2 = fuse2(d_model, channels, feature_size)

    def forward(self, hidden_states_b, hidden_states_a, diff_feat, labels):
        """
           hidden_states_b: enhanced before-temporal features from the last layer
           hidden_states_a: enhanced after-temporal features from the last layer
           diff_feat：enhanced difference features from the last layer
           labels：真实类别标签，0表示未变化，1表示变化
        """
        B, L, D = diff_feat.shape
        orig_shape = diff_feat.shape  # 【B,L,D】
        h = w = int(math.sqrt(L))

        # image-level router
        top1_idx, top1_weight, logits = self.clsgate(diff_feat)
        if labels is not None:
            flat_top1_idx = labels.view(-1)  # 【B*1】
        else:
            flat_top1_idx = top1_idx.view(-1)  # 【B*1】
        mask_change = (flat_top1_idx == 1)  # 【B*1】, bool
        mask_nochange = ~mask_change

        F_diff = diff_feat.repeat_interleave(1, dim=0)
        y = torch.empty_like(F_diff)  # 【B,L,D】

        if mask_nochange.any():
            hidden_states_fuse_1 = self.reblock1(hidden_states_b[mask_nochange], hidden_states_a[mask_nochange])
            y[mask_nochange] = self.pwff(hidden_states_fuse_1)  # 【N,L,D】

        if mask_change.any():
            hidden_states_fuse_2 = self.reblock2(hidden_states_b[mask_change], hidden_states_a[mask_change])
            y[mask_change] = self.changemoe(hidden_states_fuse_2)  # 【M,L,D】
            y[mask_change] = self.layer_norm(hidden_states_fuse_2 + y[mask_change])

        y = y.to(F_diff.dtype).view(*orig_shape)
        return y, logits  # 【B,L,D】，【B,2】
