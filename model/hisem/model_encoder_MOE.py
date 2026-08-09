import torch
from sympy.codegen.ast import continue_
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange
import clip
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModel
from model.hisem.CascadeMoE import Implement_MoE
import numpy as np
from model.hisem.Differential_Change_Attention import TripleDifferentialEnhancer


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)
            modules = list(cnn.children())[:-1]

        elif 'CLIP' in self.network:
           clip_model_type = self.network.replace('CLIP-', '')
           self.clip_model, preprocess = clip.load(clip_model_type, jit=False)
           self.clip_model = self.clip_model.to(dtype=torch.float32)

        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if "CLIP" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)

        else:
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            feat1_list = feat1_list[-4:]
            feat2_list = feat2_list[-4:]

        return img_feat_A, img_feat_B

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if 'CLIP' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-4:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True
        elif 'CLIP' not in self.network and fine_tune:
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, dropout=.1, attention_module_kwargs=None):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels
        self.heads = heads
        # position embedding
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        embed_dim = channels
        self._reset_parameters()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.moe = Implement_MoE(d_model=embed_dim, d_ff=2048, dropout=.1, feature_size=feature_size,
                                   attention_module_kwargs=attention_module_kwargs)
        self.diff_attn = TripleDifferentialEnhancer(dim=embed_dim, n_layers=n_layers, feature_size=feature_size, attention_module_kwargs=attention_module_kwargs, dropout=dropout)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3: # NLD
            b = x.shape[0]
            c = x.shape[-1]
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        # reshape back to NLD
        x = x.view(b, c, -1).transpose(-1, 1)  # NLD (b,hw,c)
        return x

    def forward(self, img_A, img_B, cls_label):
        h, w = self.h_feat, self.w_feat
        # 1. A B feature from backbone  NLD
        img_A = self.add_pos_embedding(img_A)
        img_B = self.add_pos_embedding(img_B)
        # 2. captioning
        batch, c = img_A.shape[0], img_A.shape[-1]
        img_sa1, img_sa2 = img_A, img_B
        N, L, D = img_sa1.shape

        enhanced_sa1, enhanced_sa2 = self.diff_attn(img_sa1, img_sa2)  # NLD
        enhanced_diff = enhanced_sa2 - enhanced_sa1
        feat_cap_out, cls_logits_out = self.moe(enhanced_sa1, enhanced_sa2, enhanced_diff, cls_label)
        feat_cap_out = self.layer_norm(enhanced_sa1 + enhanced_sa2 + feat_cap_out)
        return feat_cap_out, enhanced_diff, cls_logits_out


if __name__ == '__main__':
    # test
    img_A = torch.randn(16, 49, 768).cuda()
    img_B = torch.randn(16, 49, 768).cuda()
    encoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8).cuda()
    feat_cap = encoder(img_A, img_B)
    print(feat_cap.shape)
    print(feat_cap)
    print('Done')
