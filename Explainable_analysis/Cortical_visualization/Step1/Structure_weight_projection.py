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
plt.style.use('ggplot')
decord.bridge.set_bridge('torch')

device = torch.device('cuda:2')
class ContentNetwork(nn.Module):
    def __init__(self, in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=1, norm_type='ln', act_first=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
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
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
        y = self.lin1(x)
        return y.view(-1,1,4*64*64)





#/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/content_decoder/first_frame/without_res_block2_warmup_50/ContentNetwork_200.pth
#/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/content_decoder/first_frame/more_aug_block1_warmup_50
model_first = ContentNetwork(in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=2, norm_type='ln', act_first=False)
model_first.load_state_dict(torch.load('/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub02/pretrained_weights/content/ContentNetwork_75.pth'))
model_first.eval()

print('')

Lin0_par = model_first.lin0[0].weight.data.cpu().numpy().T             #[512,4500]

MLP1 = model_first.mlp[0][0].weight.data.cpu().numpy().T             #[512,512]
MLP2 = model_first.mlp[1][0].weight.data.cpu().numpy().T              #[512,512]

Lin1_1 = model_first.lin1[2].weight.data.cpu().numpy().T              #[2048,512]

#Content = np.abs(np.mean((((Lin0_par @ MLP1) @ MLP2 )@Lin1_1), axis=1))
Content = np.abs(np.mean((Lin0_par), axis=1))
min_val = np.min(Content)
max_val = np.max(Content)
# 进行归一化处理
normalized_array = ((Content - min_val) / (max_val - min_val))

mask = np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub02/activated_mask/mask_correct.npy')
mask_idx = np.nonzero(mask)[0]



path_c = '/nfs/nica-datashop/CC2017_Purdue/Subject02/video_fmri_dataset/subject2/fmri/seg1/cifti/seg1_2_Atlas.dtseries.nii'
old_cifti = nib.load(path_c)
data_c = old_cifti.get_fdata()
data = data_c.copy()
data[:10, :] = 0
for j in range(4500):
    idx = mask_idx[j]
    data[:10, idx] = normalized_array[j]


#D = np.array(data[0,:])
#print(np.nonzero(D)[0])

clipped_img = nib.Cifti2Image(data, header=old_cifti.header,nifti_header=old_cifti.nifti_header,file_map=old_cifti.file_map)
new = clipped_img.get_fdata()
#nib.save(clipped_img, '/data0/home/luyizhuo/NIPS2024实验材料/皮层可视化/Sub01_Content_first_layer.dtseries.nii')

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



#top_30_indices = np.argsort(roiweights)[-30:]
#print(np.sort(roiweights)[-20:])
# 输出索引
#print("最大的 30 个数的索引：", top_30_indices+1)
#[338          2      140     186    196   320        159 200  13 336  23 187 321 156 193 141 182 203    16  21]
#'L_V3CD', 'R_MST','R_TPOJ2','L_V4','L_V7','L_TPOJ2','R_LO3'

l_draw = np.array(['R_V1','R_MST','R_TPOJ2','R_V3','R_V4','R_V3A','R_V3B', 'R_MT','L_V1', 'R_V2', 'R_TPOJ1','L_MST','L_V2','L_V3','L_V4','L_V3A','L_V3B', 'L_MT','L_TPOJ1', 'L_TPOJ2'])
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




