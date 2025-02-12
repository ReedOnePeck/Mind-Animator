import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from functools import partial
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import matplotlib.cm as cm
import decord
import matplotlib.pyplot as plt
plt.style.use('ggplot')

scaler = StandardScaler()

device = torch.device('cuda:0')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500000):
        """
        位置编码器类的初始化函数

        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
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
        #初始化函数有两个参数:feature_size,表示词嵌入的维度; eps是一个足够小的数，防止分母为0，默认是1e-6。
        super(LayerNorm, self).__init__()
        #根据features的形状初始化两个参数张量a2，和b2，第一初始化为1张量，第二个初始化为0张量，这两个张量就是规范化层的参数。因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，使其即能满足规范化要求，又能不改变针对目标的表征，最后使用nn.parameter封装，代表他们是模型的参数
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.zeros(feature_size))
        #把eps传到类中
        self.eps = eps

    def forward(self, x):
    #输入参数x代表来自上一层的输出，在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致，接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果。
    #最后对结果乘以我们的缩放参数，即a2,*号代表同型点乘，即对应位置进行乘法操作，加上位移参b2，返回即可
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def subsequent_mask(size, mask_ratio):
    #Random Causal mask
    #1.不能看到未来的视频帧
    #2.由于视频在时间维度的高度冗余性，因此对于可见的视频帧仍要进行随机掩码
    #3.调参的时候注意，对于每一个iteration，重新随机生成一个mask，还是说对每一层都随机生成一个iteration
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    for i in range(size):
        if i >1:
            for j in range(size):
                if random.random() < 1 -mask_ratio and j!=0:
                    subsequent_mask[:,i,j] = 1

    return torch.from_numpy(subsequent_mask) == 0

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
        #h代表头数，d_model代表词嵌入的维度
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h ==0
        self.d_k = d_model // (h*16) #每个头的词向量维度
        self.h = h

        #需要四个投影层，前三个分别作用与k,q,v,最后一个用于多头注意力聚合后的投影层
        self.linears = clones(nn.Linear(d_model, 1024), 3)
        self.linears1 = nn.Linear(1024, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)   #对不同的头使用相同的mask，扩展的维度代表头数

        nbatches = query.size(0)  #batch_size

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                             for l,x in zip(self.linears, (query, key, value))]
        #先把k,q,v通过线性映射l(x)后再划分成多头，其中self.h代表头数，self.d_k代表每个头的词向量维度，-1会自动成为句子的长度
        #为了使注意力机制注意到句子中的每个词，需要让句子长度的维度和词向量维度相邻，因此使用.transpose(1,2)
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

    def forward(self, x,mask):
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
        self.tgt_embed = tgt_embed # ouput embedding module


    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model( N=2, d_model=4096*4, d_ff=768, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        BrainEncoder(out_dim=d_model, in_dim=4500, h=768, n_blocks=1, norm_type='ln', act_first=False),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model), c(position)))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def greedy_decode(first_frame,  model, src, src_mask,  max_len ,mask_ratio):
    decoder_input = first_frame
    src = torch.tensor(src)
    memory = model.encode(src, src_mask)                                        #src的数据类型为tensor
    ys = torch.ones(1, 1).fill_(1).type_as(src.data)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(decoder_input),
                           Variable(subsequent_mask(ys.size(1), mask_ratio).type_as(src.data).to(device))   )
        next_frame = (out[:,-1,:]).unsqueeze(dim=1)
        ys = torch.cat([ys,torch.ones(1, 1).fill_(1).type_as(src.data)], dim=1)
        decoder_input = torch.cat([decoder_input,next_frame] , dim=1)

    return decoder_input


model_content = make_model(2, d_model=4096*4, d_ff=768, h=4, dropout=0.2)
model_content.load_state_dict(torch.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/content_decoder/motion_content/bs_64_block_2_warmup_100/model_70.pth', map_location=torch.device('cuda:0')))
model_content.to(device)
model_content.eval()

print('')

lin0 = model_content.encoder.lin0[0].weight.data.cpu().numpy().T
mlp = model_content.encoder.mlp[0][0].weight.data.cpu().numpy().T
lin1 = model_content.encoder.lin1.weight.data.cpu().numpy().T

Content = np.abs(np.mean(((lin0 @ mlp) @ lin1 ), axis=1))
min_val = np.min(Content)
max_val = np.max(Content)
# 进行归一化处理
normalized_array = ((Content - min_val) / (max_val - min_val))

mask = np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub03/activated_mask/mask_correct.npy')
mask_idx = np.nonzero(mask)[0]



path_c = '/nfs/nica-datashop/CC2017_Purdue/Subject03/video_fmri_dataset/subject3/fmri/seg1/cifti/seg1_2_Atlas.dtseries.nii'
old_cifti = nib.load(path_c)
data_c = old_cifti.get_fdata()
data = data_c.copy()
data[:10, :] = 0
for j in range(4500):
    idx = mask_idx[j]
    data[:10, idx] = normalized_array[j]


clipped_img = nib.Cifti2Image(data, header=old_cifti.header,nifti_header=old_cifti.nifti_header,file_map=old_cifti.file_map)
new = clipped_img.get_fdata()
nib.save(clipped_img, '/data0/home/luyizhuo/NIPS2024实验材料/皮层可视化/sub03_consist.dtseries.nii')

"""
path_label = '/data0/home/luyizhuo/NIPS2024实验材料/皮层可视化/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
label_cifti = nib.load(path_label)
data_label = label_cifti.get_fdata()[0]

data1 = data[0, :59412]
#data1norm = data1/np.sum(data1)
roiweights = np.zeros((1,360))[0]

for roi in range(360):
    #count = np.sum(data_label == roi)
    a = np.count_nonzero(data1[data_label == (roi+1)])
    if a !=0 :
        roiweights[roi] = np.sum(data1[data_label == (roi+1)]) / np.count_nonzero(data1[data_label == (roi+1)])
    else:
        roiweights[roi] = np.sum(data1[data_label == (roi+1)])


#top_15_indices = np.argsort(roiweights)[-15:]
#print(np.sort(roiweights)[-15:])
# 输出索引
#print("最大的 15 个数的索引：", top_15_indices)
# 'R_7PC', 'R_TPOJ3', 'L_LIPv','R_V3A' ,'L_V3A' ,'R_STV', 'L_TPOJ2', 'R_V4', 'R_V3','R_V2', 'L_V4', 'L_V3','L_V2','R_V1','L_V1'
#[0.06536762 0.06546165 0.06618924 0.0700056  0.07099683 0.07164565
# 0.07491737 0.07769521 0.08171014 0.08655089 0.09212514 0.10010832
# 0.10784062 0.10924793 0.10959705]


#bar_width = 0.8



l_draw = np.array(['R_V1','R_MST','R_V2','R_V3','R_V4','R_V3A','R_V3B', 'R_MT','R_TPOJ1', 'R_TPOJ2', 'L_V1','L_MST','L_V2','L_V3','L_V4','L_V3A','L_V3B', 'L_MT','L_TPOJ1', 'L_TPOJ2'])

idx = np.array([1,2,4,5,6,13,19,23,139,140,181,182,184,185,186,193,199,203,319,320])-1


#idx = map(lambda x:x-1,idx)
data_draw = roiweights[idx]

data = np.sort(data_draw)
print('权值为：')
print(data)
labels_idx = np.argsort(data_draw)
labels_draw = l_draw[labels_idx]



norm = plt.Normalize(vmin=data.min(), vmax=data.max())

bar_positions = range(len(data))  # y 轴的位置
colors = cm.YlOrRd_r(norm(data))

plt.barh(bar_positions, data, color=colors)

custom_yticks = labels_draw.tolist()

plt.yticks(bar_positions, custom_yticks, fontsize=10, fontname='DejaVu Sans')
plt.xticks(fontsize=12, fontname='DejaVu Sans')
plt.xlabel('weight proportion',fontsize=12, fontname='DejaVu Sans')


# 显示图表
plt.show()


"""











