import math
import numpy
import numpy as np
from tqdm import tqdm
import torch
import imageio
from einops import rearrange
import os
import imageio.v3 as iio

def psnr(img1, img2):
    #img:0~255
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = numpy.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1



gt_video_sub1 = rearrange(torch.tensor(np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/test_gt_npy/test_groundtruth.npy')), 'b c f h w -> b f h w c').numpy()
#print(gt_video_sub1.shape)       #1200,8,512,512,3


S = []
for i in tqdm(range(1200)):
    ssim = []
    with imageio.get_reader('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub01/recons_only/test{}.gif'.format(i+1)) as gif:
        for j, frame in enumerate(gif):
            f_recons = np.array(frame )
            #print(f_recons.shape)
            f_gt = gt_video_sub1[i,j,:,:,:] * 255.
            #print(f_gt.shape)
            c = psnr(f_recons, f_gt)
            ssim.append(c)

    s = np.mean(ssim)
    S.append(s)

print(np.mean(S))
print(np.std(S))
"""

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
        c = psnr(recons, gt)
        ssim.append(c)
    s = np.mean(ssim)
    S.append(s)

print(np.mean(S))
print(np.std(S))


"""





















