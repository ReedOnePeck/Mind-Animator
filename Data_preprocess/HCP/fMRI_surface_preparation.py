import numpy as np
import itertools
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


#[1,2,4,5,6,13,19,20,21,23,139,140,181,182,184,185,186,193,199,200,201,203,319,320]
ROI_Keys = np.array([1,2,4,5,6,13,19,20,21,23,139,140,181,182,184,185,186,193,199,200,201,203,319,320])
ROI_masks = nib.load('/data0/home/luyizhuo/HCP预处理/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii').get_fdata()[0]
ROI_idx = np.argwhere(np.in1d(ROI_masks, ROI_Keys)).reshape(-1)


clips = np.load('/data0/home/luyizhuo/HCP预处理/clip_times_24.npy', allow_pickle=True).item()
idx = 0
solid_clips = clips['{}'.format(idx+1)]
start_times = []
for c in range(len(solid_clips)):
    start_times.append(np.arange((solid_clips[c, 0] / 24)+4, (solid_clips[c, 1] / 24)+4).astype('int'))
start_times = np.array(list(itertools.chain(*start_times)))

sub104416 = np.array(nib.load('/data0/home/luyizhuo/HCP_surface/104416/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/tfMRI_MOVIE1_7T_AP_Atlas.dtseries.nii').get_fdata())[start_times, :]
fMRI = sub104416[:,ROI_idx]
fMRI = (fMRI - np.min(fMRI, axis=0)) / (np.max(fMRI, axis=0) - np.min(fMRI, axis=0))

Img = np.load('/data0/home/luyizhuo/HCP预处理/CLIP_img_embeddings.npy')[:728,:]

print(fMRI.shape)
print(Img.shape)

#RDM
corr1 = np.corrcoef(fMRI[:,:])
plt.imshow(corr1, cmap='Blues')
plt.title('fMRI')
plt.colorbar()
plt.show()


corr2 = np.corrcoef(Img[:,:])
plt.imshow(corr2, cmap='Blues')
plt.title('CLIP')
plt.colorbar()
plt.show()

correlation_matrix, p_value = spearmanr(corr1, corr2, axis=None)
print(correlation_matrix)
print(p_value)

