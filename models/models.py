import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math, copy
from einops import rearrange, reduce, repeat
from timm.models.vision_transformer import PatchEmbed, Mlp


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size, mask_ratio):
    #Random Causal mask
    #1.不能看到未来的视频帧
    #2.由于视频在时间维度的高度冗余性，因此对于可见的视频帧仍要进行随机掩码
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    for i in range(size):
        if i >1:
            for j in range(size):
                if random.random() < 1 -mask_ratio and j!=0:
                    subsequent_mask[:,i,j] = 1

    return torch.from_numpy(subsequent_mask) == 0

def modulate(x, shift, scale, N):

    T, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b n) t m-> b (t n) m',b=B,t=T,n=N,m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b n) t m',b=B,t=T,n=N,m=M)
    return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Temporal_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None) :
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Spatial_Attention(nn.Module):
    def __init__(self, h, d_model, dropout=0.15):
        super(Spatial_Attention, self).__init__()
        assert d_model % h ==0
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linears1 = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self,query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        #a = self.linears1(query)
        #b = self.linears1(key)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                             for l,x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask = mask, dropout = self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears1(x)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.linear(self.norm_final(x))
        return x


class CMG_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,  num_frames=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.spatial_attention = Spatial_Attention(num_heads, hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_attention = Temporal_Attention(dim = hidden_size , num_heads =  num_heads , qkv_bias = True)
        self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, spatial_info, temporal_info, mask = None):
        #shift_mlp, scale_mlp = self.adaLN_modulation(temporal_info).chunk(2, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        #temporal_attention

        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        res_temporal = self.temporal_attention(x, mask)
        #res_temporal = modulate(self.temporal_norm1(res_temporal), shift_mlp, scale_mlp, N)
        res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)
        #res_temporal = self.temporal_fc(res_temporal)
        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)

        #res_temporal = self.spatial_attention(self.norm1(x), temporal_info.unsqueeze(1).expand(-1, T, -1),
        #                              temporal_info.unsqueeze(1).expand(-1, T, -1))
        x = x + res_temporal

        #spatial_attention
        attn = self.spatial_attention(self.norm1(x), spatial_info.unsqueeze(1).expand(-1, T, -1), spatial_info.unsqueeze(1).expand(-1, T, -1))
        #attn = self.spatial_attention(self.norm1(x), self.norm1(x), self.norm1(x))
        #attn = rearrange(attn, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        #attn = modulate(self.temporal_norm1(attn), shift_mlp, scale_mlp, N)
        #attn = rearrange(attn, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)
        x = x + attn

        mlp = self.mlp(self.norm2(x))
        x = x + mlp
        return x


class CMG(nn.Module):
    def __init__(
            self,
            input_size=64,
            patch_size=4,
            in_channels=4,
            voxel_size = 4500,
            hidden_size=512,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            num_frames=7
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.frame_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)     #检查一下out_channel是否等于in_channel，需要对应修改finallayer的参数
        num_patches = self.frame_embedder.num_patches

        self.fMRI_to_spatial = nn.Sequential(nn.SiLU(), nn.Linear(voxel_size, hidden_size, bias=True))
        self.fMRI_to_temporal = nn.Sequential(nn.SiLU(), nn.Linear(voxel_size, hidden_size, bias=True))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)

        self.blocks = nn.ModuleList([CMG_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_frames=self.num_frames) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.frame_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.frame_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.frame_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.frame_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, fMRI, x,  mask = None):
        """
        x: (N, C, H, W) tensor of spatial inputs (latent representations of images)  N = batch_size * time
        """
        B, T, C, W, H = x.shape  # 32 7 4 64 64
        x = x.contiguous().view(-1, C, W, H)
        x = self.frame_embedder(x) + self.pos_embed  # (N, P, D), where P = H * W / patch_size ** 2

        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        ## Resizing time embeddings in case they don't match
        x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)

        spatial_info = self.fMRI_to_spatial(fMRI)  # B, D
        temporal_info = self.fMRI_to_temporal(fMRI)

        for block in self.blocks:         #dimension_in : (b t) n m  -->  spatial -->  temporal     dimension_out :(b n) t m
            x = block(x, spatial_info, temporal_info, mask)  # (N, T, D)
        x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x


def custom_mask(length, idx):
    # 创建一个全为1的下三角矩阵
    mask = np.tril(np.ones((length, length)), 0)

    # 根据idx设置下三角的部分可见性
    for i in range(idx, length):
        mask[i, :] = 0  # 将 idx 之后的行设置为不可见（全为0）

    return torch.from_numpy(mask) == 0

def greedy_decode(first_frame,  model, fMRI,  max_len, greedy_Mask):
    model.eval()
    with torch.no_grad():
        decoder_input = first_frame
        for i in range(max_len):
            mask = greedy_Mask[i]
            out = model.forward(fMRI, decoder_input,  mask)
            next_frame = (out[:, i, :,:,:]).unsqueeze(dim=1)
            if i + 1 != max_len:
                decoder_input[:,i+1:i+2,:,:,:] = next_frame
                del out, next_frame
                torch.cuda.empty_cache()
            else:
                decoder_input = torch.cat([decoder_input, next_frame], dim=1)
                del out, next_frame
                torch.cuda.empty_cache()
    return decoder_input





"""
cmg = CMG()
mask = subsequent_mask(size = 7, mask_ratio = 0.6)
fMRI = torch.rand(64,4500)
latent_video = torch.rand(64, 7,4,64,64)
out_put = cmg(latent_video, fMRI, mask)
print(out_put.shape)
"""

"""
x = torch.rand(64, 8, 512)        # Query 序列
mask = subsequent_mask(size = 8, mask_ratio = 0.6)
cross_attention = Temporal_Attention(dim = 512)
output = cross_attention(x,mask)
print('')
"""