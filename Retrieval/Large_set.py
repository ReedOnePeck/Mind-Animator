import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import decord
decord.bridge.set_bridge('torch')

device = torch.device('cuda:2')
seed = 42




def top_k_accuracy(data, queries, labels, k=1):
    correct = 0
    total = len(queries)

    for idx, query in enumerate(queries):
        # 计算查询和数据集中每个点的余弦相似度
        sim = cosine_similarity([query], data).flatten()
        # 根据相似度排序，找出top-k的索引
        top_k_indices = sim.argsort()[-k:][::-1]

        # 检查正确的索引是否在top-k中
        if labels[idx] in top_k_indices:
            correct += 1

    return correct / total

class BrainNetwork(nn.Module):
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


    def forward(self, x):
        x = self.lin0(x)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        return x


subj_ID = 3
epoch = 40
model = BrainNetwork(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=2, norm_type='ln', act_first=False)
model.load_state_dict(torch.load('/data0/home/luyizhuo/NIPS2024实验材料/Retreive/CC2017/sub{}/'.format(subj_ID) + 'BrainNetwork_{}.pth'.format(epoch)))
model.to(device)
model.eval()

test_src_path = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/activated_voxels_PCA/masked4500_test_data.npy'.format(subj_ID)
test_src = torch.tensor(np.load(test_src_path), dtype=torch.float32).to(device)
semantic = model(torch.tensor(test_src)).cpu().detach().numpy()

CC2017_CLIP_emb = np.load('/nfs/nica-datashop/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_img_embeddings_CLIP_512.npy')
HCP_emb = np.load('/data0/home/luyizhuo/HCP预处理/CLIP_img_embeddings.npy')


data = np.concatenate([CC2017_CLIP_emb, HCP_emb], axis=0)
queries = semantic
labels = np.arange(1200)

top_10_acc = top_k_accuracy(data, queries,labels, k=10)
top_100_acc = top_k_accuracy(data, queries, labels, k=100)

print(f"Top-10 Accuracy: {top_10_acc:.4f}")
print(f"Top-100 Accuracy: {top_100_acc:.4f}")


