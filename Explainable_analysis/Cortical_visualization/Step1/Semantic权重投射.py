import numpy as np
import torch
import torch.nn as nn
from functools import partial
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import nibabel as nib
import matplotlib.cm as cm
import decord
import matplotlib.pyplot as plt
decord.bridge.set_bridge('torch')
plt.style.use('ggplot')
device = torch.device('cuda:2')

class Semantic_Network(nn.Module):
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

    def forward(self, x ):
        x = self.lin0(x)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        y = self.lin1(x)

        return x, y.view(-1, 19, 768)


model = Semantic_Network(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False)
model.load_state_dict(torch.load('/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub03/pretrained_weights/semantic/BrainNetwork_30.pth'))
model.eval()
Lin0_par = model.lin0[0].weight.data.cpu().numpy().T             #[512,4500]

MLP1 = model.mlp[0][0].weight.data.cpu().numpy().T             #[512,512]
MLP2 = model.mlp[1][0].weight.data.cpu().numpy().T              #[512,512]
MLP3 = model.mlp[2][0].weight.data.cpu().numpy().T               #[512,512]

Lin1_1 = model.lin1[2].weight.data.cpu().numpy().T              #[2048,512]
Lin1_2 = model.lin1[5].weight.data.cpu().numpy().T
Lin1_3 = model.lin1[8].weight.data.cpu().numpy().T

Semantic = np.abs(np.mean((((((Lin0_par @ MLP1) @ MLP2 )@ MLP3)@Lin1_1) @Lin1_2)@Lin1_3, axis=1))
#Semantic = np.abs(np.mean(Lin0_par, axis=1))
#print(np.mean(Semantic, axis=1).shape)
min_val = np.min(Semantic)
max_val = np.max(Semantic)

# 进行归一化处理
normalized_array = ((Semantic - min_val) / (max_val - min_val))


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
nib.save(clipped_img, '/data0/home/luyizhuo/NIPS2024实验材料/皮层可视化/sub03_Semantic.dtseries.nii')



path_label = '/data0/home/luyizhuo/NIPS2024实验材料/皮层可视化/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
label_cifti = nib.load(path_label)

data_label = label_cifti.get_fdata()[0]  #从1到360的编号，每一个编号对应一个脑区，但是有5万多个数，每一个数代表一个体素
print(np.max(data_label))

data1 = data[0, :59412]
#data1norm = data1/np.sum(data1)
roiweights = np.zeros((1,360))[0]

for roi in range(360):
    a = np.count_nonzero(data1[data_label == (roi+1)])
    if a !=0 :
        roiweights[roi] = np.sum(data1[data_label == (roi+1)]) / np.count_nonzero(data1[data_label == (roi+1)])
    else:
        roiweights[roi] = np.sum(data1[data_label == (roi+1)])

#top_15_indices = np.argsort(roiweights)[-15:]

#print(np.sort(roiweights)[-15:])
# 输出索引
#print("最大的 15 个数的索引：", top_15_indices+1)
#[ 321    140     28     47       5     4      13    193  320   1     6     185   186   184  181]
#L_TPOJ3,R_TPOJ2,R_STV,R_7PC,  R_V3,  R_V2,  R_V3A, L_V3A,L_TPOJ2, R_V1, R_V4, L_V3, L_V4, L_V2, L_V1
#[0.01825826 0.01878116 0.01893226 0.0201877  0.02604493 0.02658893
# 0.02799935 0.02966597 0.03198913 0.03333543 0.03821684 0.04572055
# 0.05129875 0.05680607 0.0881571 ]
#low:   hight:
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
plt.yticks(bar_positions, custom_yticks, fontsize=10, fontname='sans-serif')
plt.xticks(fontsize=14, fontname='sans-serif')
plt.xlabel('weight proportion',fontsize=14, fontname='sans-serif')


# 显示图表
plt.show()





