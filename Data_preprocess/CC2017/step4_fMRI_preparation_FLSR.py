import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
pca = PCA(n_components=3000)
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()
#---------------------------------------------------------------------------------------------------------------------------------------------------
fMRI_volumes_root = '/nfs/nica-datashop/CC2017_Purdue/Subject0{}/video_fmri_dataset/subject{}/fmri/'
raw_train_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/voxels/'
averaged_train_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/voxels_avg/'
# raw_test_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/voxels/'
averaged_test_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/voxels_avg/'

mask_save_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/activated_mask/'
masked_trn_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/activated_voxels_avg/'
masked_trn_data_PCA = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/activated_voxels_PCA/'
masked_test_data_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/activated_voxels_avg/'
masked_test_data_PCA = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/activated_voxels_PCA/'
#---------------------------------------------------------fMRI_volume---------------------------------------------------------------------------
#去除fMRI信号中的趋势项
def amri_sig_detrend(its, polyorder):
    if polyorder < 0:
        polyorder = 0

    nt = len(its)
    t = np.linspace(0, 1, nt)

    # 多项式拟合趋势项
    trend = np.polyfit(t, its, polyorder)
    trend_line = np.polyval(trend, t)

    # 去趋势化后的信号
    detrended_signal = its - trend_line

    return detrended_signal, trend_line

#去除趋势项之后做zsore
def zscore_normalize(matrix):
    mean_values = np.mean(matrix, axis=0)
    std_dev_values = np.std(matrix, axis=0)
    zscore_matrix = (matrix - mean_values) / std_dev_values

    return zscore_matrix



for subj_ID in tqdm(range(1, 4)):
    for seg_ID in tqdm(range(1, 19)):
        fMRI_volumes_path_1 = fMRI_volumes_root.format(subj_ID,
                                                       subj_ID) + 'seg{}/cifti/seg{}_1_Atlas.dtseries.nii'.format(
            seg_ID, seg_ID)
        fMRI_volumes_path_2 = fMRI_volumes_root.format(subj_ID,
                                                       subj_ID) + 'seg{}/cifti/seg{}_2_Atlas.dtseries.nii'.format(
            seg_ID, seg_ID)

        fMRI_session1 = nib.load(fMRI_volumes_path_1).get_fdata()[[1 + j for j in range(240)], :59412]
        fMRI_session1_detrended = np.zeros((fMRI_session1.shape[0], fMRI_session1.shape[1]))
        for voxel_idx in range(fMRI_session1.shape[1]):
            detrended_voxel, _ = amri_sig_detrend(its=fMRI_session1[:,voxel_idx], polyorder=4)
            fMRI_session1_detrended[:,voxel_idx] = detrended_voxel
        fMRI_session1_zscored = zscore_normalize(fMRI_session1_detrended)

        if not os.path.exists(raw_train_data_root.format(subj_ID)):
            os.makedirs(raw_train_data_root.format(subj_ID))
        np.save(raw_train_data_root.format(subj_ID) + 'seg{}_1.npy'.format(seg_ID), fMRI_session1_zscored)
        #==============================================================================================================

        fMRI_session2 = nib.load(fMRI_volumes_path_2).get_fdata()[[1 + j for j in range(240)], :59412]
        fMRI_session2_detrended = np.zeros((fMRI_session2.shape[0], fMRI_session2.shape[1]))
        for voxel_idx in range(fMRI_session2.shape[1]):
            detrended_voxel, _ = amri_sig_detrend(its=fMRI_session2[:, voxel_idx], polyorder=4)
            fMRI_session2_detrended[:, voxel_idx] = detrended_voxel
        fMRI_session2_zscored = zscore_normalize(fMRI_session2_detrended)
        np.save(raw_train_data_root.format(subj_ID) + 'seg{}_2.npy'.format(seg_ID), fMRI_session2_zscored)

        # ============================================================================================================
        if not os.path.exists(averaged_train_data_root.format(subj_ID)):
            os.makedirs(averaged_train_data_root.format(subj_ID))
        clip = (fMRI_session1_zscored + fMRI_session2_zscored) / 2
        np.save(averaged_train_data_root.format(subj_ID) + 'seg{}.npy'.format(seg_ID), clip)
print("Training_data_Done")


for subj_ID in tqdm(range(1, 4)):
    # 对第一个测试数据片段进行处理
    session_data = []
    for session_ID in range(1, 11):
        fMRI_volumes_path = fMRI_volumes_root.format(subj_ID,
                                                     subj_ID) + 'test1/cifti/test1_{}_Atlas.dtseries.nii'.format(
            session_ID)
        fMRI = nib.load(fMRI_volumes_path).get_fdata()[[1 + j for j in range(240)], :59412]
        fMRI_detrended = np.zeros((fMRI.shape[0], fMRI.shape[1]))
        for voxel_idx in range(fMRI.shape[1]):
            detrended_voxel, _ = amri_sig_detrend(its=fMRI[:, voxel_idx], polyorder=4)
            fMRI_detrended[:, voxel_idx] = detrended_voxel
        fMRI_zscored = zscore_normalize(fMRI_detrended)
        session_data.append(fMRI_zscored)

    session_mean = np.mean(session_data, axis=0)

    if not os.path.exists(averaged_test_data_root.format(subj_ID)):
        os.makedirs(averaged_test_data_root.format(subj_ID))
    np.save(averaged_test_data_root.format(subj_ID) + 'test1.npy', session_mean)

    # 对第2-5个测试数据片段进行处理
    for seg_ID in tqdm(range(2, 6)):
        if seg_ID == 2 and subj_ID == 3:
            test = []
            for session_ID in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
                fMRI_volumes_path = fMRI_volumes_root.format(subj_ID,
                                                             subj_ID) + 'test{}/cifti/test{}_{}_Atlas.dtseries.nii'.format(
                    seg_ID, seg_ID, session_ID)
                fMRI = nib.load(fMRI_volumes_path).get_fdata()[[2 + j for j in range(240)], :59412]
                fMRI_detrended = np.zeros((fMRI.shape[0], fMRI.shape[1]))
                for voxel_idx in range(fMRI.shape[1]):
                    detrended_voxel, _ = amri_sig_detrend(its=fMRI[:, voxel_idx], polyorder=4)
                    fMRI_detrended[:, voxel_idx] = detrended_voxel
                fMRI_zscored = zscore_normalize(fMRI_detrended)
                test.append(fMRI_zscored)

            clip = np.mean(test, axis=0)
            np.save(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID), clip)
        else:
            session_data = []
            for session_ID in range(1, 11):
                fMRI_volumes_path = fMRI_volumes_root.format(subj_ID,
                                                             subj_ID) + 'test{}/cifti/test{}_{}_Atlas.dtseries.nii'.format(
                    seg_ID, seg_ID, session_ID)
                fMRI = nib.load(fMRI_volumes_path).get_fdata()[[2 + j for j in range(240)], :59412]
                fMRI_detrended = np.zeros((fMRI.shape[0], fMRI.shape[1]))
                for voxel_idx in range(fMRI.shape[1]):
                    detrended_voxel, _ = amri_sig_detrend(its=fMRI[:, voxel_idx], polyorder=4)
                    fMRI_detrended[:, voxel_idx] = detrended_voxel
                fMRI_zscored = zscore_normalize(fMRI_detrended)
                session_data.append(fMRI_zscored)

            session_mean = np.mean(session_data, axis=0)
            np.save(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID), session_mean)

print("Test_data_Done")

#---------------------------------------------------------fMRI_mask---------------------------------------------------------------------------
def intra_subj_corr(subj_ID,seg_ID):
    a = np.zeros(59412)
    fMRI_session1 = np.load(raw_train_data_root.format(subj_ID) + 'seg{}_1.npy'.format(seg_ID))
    fMRI_session2 = np.load(raw_train_data_root.format(subj_ID) + 'seg{}_2.npy'.format(seg_ID))
    for j in range(59412):
        data_session1 = fMRI_session1[:,j]
        data_session2 = fMRI_session2[:,j]
        corr_matrix = np.corrcoef(data_session1, data_session2)
        corr_coefficient = corr_matrix[0, 1]
        a[j] = corr_coefficient
    z_transformed = 0.5 * np.log((1 + a) / (1 - a))
    return z_transformed  #(59412,)

def min_indices(arr,num):
    sorted_indices = sorted(range(len(arr)), key=lambda x: arr[x])
    min_indices = sorted_indices[:num]
    return min_indices

def visual_activate_mask(subj_ID, n = None):
    mask = np.zeros(59412)
    p_values = np.zeros(59412)
    all_segments = np.array([intra_subj_corr(subj_ID,seg_ID) for seg_ID in range(1,19)])   #(18,59412)
    for j in range(59412):
        t_statistic, p_value = stats.ttest_1samp(all_segments[:, j], 0)
        p_values[j] = p_value

    # 进行FDR校正
    rejected, adjusted_p_values, _, _ = multipletests(p_values, alpha=0.01, method='bonferroni')

    # 输出校正后的p值和拒绝假设的结果
    idxs = min_indices(arr = adjusted_p_values, num = n)
    for idx in idxs:
        mask[idx] = 1

    """
    for idx,rej in enumerate(rejected):
        if rej:
            mask[idx] = 1
    """
    if not os.path.exists(mask_save_root.format(subj_ID)):
        os.makedirs(mask_save_root.format(subj_ID))
    np.save(mask_save_root.format(subj_ID) + 'mask_correct.npy',mask)
    return mask

"""
for subj_ID in range(1,4):
    mask = visual_activate_mask(subj_ID, 1500)
    mask_idx = np.nonzero(mask)
    trn_data = np.concatenate( [np.load(averaged_train_data_root.format(subj_ID) + 'seg{}.npy'.format(seg_ID)) for seg_ID in range(1,19)],axis = 0 )
    masked_trn_data = np.squeeze(trn_data[:, mask_idx])
    data_trn = scaler1.fit_transform(masked_trn_data)
    np.save(masked_trn_data_PCA.format(subj_ID) + 'masked1500_trn_data.npy', data_trn)
    print('被试{}的训练集数据PCA完毕'.format(subj_ID))

    test_data = np.concatenate([np.load(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID)) for seg_ID in range(1, 6)],axis=0)
    masked_test_data = np.squeeze(test_data[:, mask_idx])
    data_test = scaler2.fit_transform(masked_test_data)

    np.save(masked_test_data_PCA.format(subj_ID) + 'masked1500_test_data.npy', data_test)
    print('被试{}的测试集数据PCA完毕'.format(subj_ID))

for subj_ID in range(1,4):
    mask = visual_activate_mask(subj_ID, 3000)
    mask_idx = np.nonzero(mask)
    trn_data = np.concatenate( [np.load(averaged_train_data_root.format(subj_ID) + 'seg{}.npy'.format(seg_ID)) for seg_ID in range(1,19)],axis = 0 )
    masked_trn_data = np.squeeze(trn_data[:, mask_idx])
    data_trn = scaler1.fit_transform(masked_trn_data)
    np.save(masked_trn_data_PCA.format(subj_ID) + 'masked3000_trn_data.npy', data_trn)
    print('被试{}的训练集数据PCA完毕'.format(subj_ID))

    test_data = np.concatenate([np.load(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID)) for seg_ID in range(1, 6)],axis=0)
    masked_test_data = np.squeeze(test_data[:, mask_idx])
    data_test = scaler2.fit_transform(masked_test_data)

    np.save(masked_test_data_PCA.format(subj_ID) + 'masked3000_test_data.npy', data_test)
    print('被试{}的测试集数据PCA完毕'.format(subj_ID))
"""


for subj_ID in range(1,4):
    mask = visual_activate_mask(subj_ID, 4500)
    mask_idx = np.nonzero(mask)
    trn_data = np.concatenate( [np.load(averaged_train_data_root.format(subj_ID) + 'seg{}.npy'.format(seg_ID)) for seg_ID in range(1,19)],axis = 0 )
    masked_trn_data = np.squeeze(trn_data[:, mask_idx])
    data_trn = scaler1.fit_transform(masked_trn_data)

    np.save(masked_trn_data_PCA.format(subj_ID) + 'masked4500_trn_data.npy', data_trn)
    print('被试{}的训练集数据处理完毕'.format(subj_ID))

    test_data = np.concatenate([np.load(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID)) for seg_ID in range(1, 6)],axis=0)
    masked_test_data = np.squeeze(test_data[:, mask_idx])
    data_test = scaler2.fit_transform(masked_test_data)

    np.save(masked_test_data_PCA.format(subj_ID) + 'masked4500_test_data.npy', data_test)
    print('被试{}的测试集数据处理完毕'.format(subj_ID))


"""

for subj_ID in range(1,4):
    mask = visual_activate_mask(subj_ID)
    mask_idx = np.nonzero(mask)
    trn_data = np.concatenate( [np.load(averaged_train_data_root.format(subj_ID) + 'seg{}.npy'.format(seg_ID)) for seg_ID in range(1,19)],axis = 0 )
    masked_trn_data = np.squeeze(trn_data[:, mask_idx])
    data_trn = scaler1.fit_transform(masked_trn_data)
    pca.fit(data_trn)
    PCA_trn_data = pca.transform(data_trn)

    if not os.path.exists(masked_trn_data_root.format(subj_ID)):
        os.makedirs(masked_trn_data_root.format(subj_ID))
    if not os.path.exists(masked_trn_data_PCA.format(subj_ID)):
        os.makedirs(masked_trn_data_PCA.format(subj_ID))

    np.save(masked_trn_data_root.format(subj_ID) + 'masked_trn_data_zscore.npy',data_trn)
    np.save(masked_trn_data_PCA.format(subj_ID) + 'PCA_trn_data.npy', PCA_trn_data)
    print('被试{}的训练集数据PCA完毕'.format(subj_ID))

    test_data = np.concatenate([np.load(averaged_test_data_root.format(subj_ID) + 'test{}.npy'.format(seg_ID)) for seg_ID in range(1, 6)],axis=0)
    masked_test_data = np.squeeze(test_data[:, mask_idx])
    data_test = scaler2.fit_transform(masked_test_data)
    PCA_test_data = pca.transform(data_test)

    if not os.path.exists(masked_test_data_root.format(subj_ID)):
        os.makedirs(masked_test_data_root.format(subj_ID))
    if not os.path.exists(masked_test_data_PCA.format(subj_ID)):
        os.makedirs(masked_test_data_PCA.format(subj_ID))

    np.save(masked_test_data_root.format(subj_ID) + 'masked_test_data_zscore.npy', data_test)
    np.save(masked_test_data_PCA.format(subj_ID) + 'PCA_test_data.npy', PCA_test_data)
    print('被试{}的测试集数据PCA完毕'.format(subj_ID))

"""




