from skimage.metrics import structural_similarity as SSIM
import torch
import imageio
from tqdm import tqdm
from einops import rearrange
import numpy as np
import os
import imageio.v3 as iio


"""
gt_video_sub1 = rearrange(torch.tensor(np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/test_gt_npy/test_groundtruth.npy')), 'b c f h w -> b f h w c').numpy()
print(gt_video_sub1.shape)       #1200,8,512,512,3


S = []
for i in tqdm(range(1200)):
    ssim = []
    with imageio.get_reader('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub01/recons_only/test{}.gif'.format(i+1)) as gif:
        for j, frame in enumerate(gif):
            f_recons = np.array(frame )
            #print(f_recons.shape)
            f_gt = gt_video_sub1[i,j,:,:,:] * 255.  #img:0~255
            #print(f_gt.shape)
            c = SSIM(f_recons, f_gt, data_range=255, channel_axis=-1)
            ssim.append(c)

    s = np.mean(ssim)
    S.append(s)

print(np.mean(S))
print(np.std(S))
"""


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def ssim_score_only(
                pred_videos: np.array,
                gt_videos: np.array,
                **kwargs
                ):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(ssim_metric(pred, gt))
    return np.mean(scores), np.std(scores)

import torch.nn.functional as F

def mse_metric(img1, img2):
    return F.mse_loss(torch.FloatTensor(img1/255.0), torch.FloatTensor(img2/255.0), reduction='mean').item()

def ssim_metric(img1, img2):
    return SSIM(img1, img2, data_range=255, channel_axis=-1)



gt_list = []
pred_list = []
for i  in range(1200):
    gif = iio.imread(os.path.join('/data0/home/luyizhuo/NIPS2024实验材料/Chen/sub3/sample-all', f'test{i+1}.gif'), index=None)
    gt, pred = np.split(gif, 2, axis=2)
    #print(gt)
    gt_list.append(gt)
    pred_list.append(pred)

gt_list = np.stack(gt_list)
pred_list = np.stack(pred_list)
print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')


S = []
for i in tqdm(range(1200)):
    ssim = []
    for j in range(6):
        recons = pred_list[i,j,:,:,:]
        gt = gt_list[i,j,:,:,:]
        c = SSIM(recons, gt, data_range=255, channel_axis=-1)
        ssim.append(c)
    s = np.mean(ssim)
    S.append(s)

print(np.mean(S))
print(np.std(S))


"""
for i in tqdm(range(pred_list.shape[1])):
    # ssim scores
    ssim_scores, std = ssim_score_only(pred_list[:, i], gt_list[:, i])
    print(f'ssim score: {ssim_scores}, std: {std}')
"""














