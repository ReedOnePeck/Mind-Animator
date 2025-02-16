import torch.nn as nn
import numpy as np
import torch
import random
import torch.nn.functional as F
import math, copy
from functools import partial
from torch.autograd import Variable

class Semantic_decoder(nn.Module):
    def __init__(self, in_dim=1500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm,
                                                                                              normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.n_blocks = n_blocks
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.3),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])

        self.lin1 = nn.Sequential(
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, out_dim),

        )

    def forward(self, x, y_CLIP_t=None, y_CLIP_i=None):
        x = self.lin0(x)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        y = self.lin1(x)

        logit_scale = self.logit_scale.exp()
        fMRI_embeds = x / x.norm(p=2, dim=-1, keepdim=True)
        image_embeds = y_CLIP_i / y_CLIP_i.norm(p=2, dim=-1, keepdim=True)
        text_embeds = y_CLIP_t / y_CLIP_t.norm(p=2, dim=-1, keepdim=True)

        logits_per_fMRI_t = torch.matmul(fMRI_embeds, text_embeds.t()) * logit_scale
        logits_per_fMRI_i = torch.matmul(fMRI_embeds, image_embeds.t()) * logit_scale

        return x, y.view(-1, 19, 768), logits_per_fMRI_t, logits_per_fMRI_i


#=========================================================================================================================

class Strcuture_decoder(nn.Module):
    def __init__(self, in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=1, norm_type='ln', act_first=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm,
                                                                                              normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.n_blocks = n_blocks
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.Dropout(0.2),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.Dropout(0.2)
            ) for _ in range(self.n_blocks)
        ])


        self.lin1 = nn.Sequential(
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, out_dim)
        )

    def forward(self, x ):
        x = self.lin0(x)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        y = self.lin1(x)

        return y.view(-1,1,4*64*64)


#=========================================================================================================================

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model):
        super(Embeddings, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def subsequent_mask(size, mask_ratio):
    #Random Causal mask
    #1.不能看到未来的视频帧
    #2.由于视频在时间维度的高度冗余性，因此对于可见的视频帧仍要进行随机掩码
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    for i in range(size):
        if i >1:
            for j in range(size):
                if random.random() < 1 -mask_ratio and j!=0:
                    subsequent_mask[:,i,j] = 1

    return torch.from_numpy(subsequent_mask) == 0

#--------------------------------------------------------------------------------------------------------------------------------------------
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0,-1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.15):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h ==0
        self.d_k = d_model // (h*16)
        self.h = h

        self.linears = clones(nn.Linear(d_model, 1024), 3)
        self.linears1 = nn.Linear(1024, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                             for l,x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears1(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.15):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        sublayer_out = sublayer(x)
        sublayer_out = self.dropout(sublayer_out)
        x_norm = x + self.norm(sublayer_out)
        return x_norm

#Encoder-------------------------------------------------------------------------------------------------------------------------
class BrainEncoder(nn.Module):
    def __init__(self, out_dim=4096*4, in_dim=4500, h=768, n_blocks=3, norm_type='ln', act_first=False ):
        super(BrainEncoder,self).__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm,
                                                                                              normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.3),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks

    def forward(self, x, mask):
        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        return x

# Decoder----------------------------------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer,self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self,x ,memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x, self.feed_forward)



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder,  tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed


    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_CMG_model( N=2, d_model=4096*4, d_ff=768, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        BrainEncoder(out_dim=d_model, in_dim=4500, h=768, n_blocks=1, norm_type='ln', act_first=False),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model), c(position)))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


