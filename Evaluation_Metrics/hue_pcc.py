
import cv2
import numpy as np
from scipy import stats
import os
import imageio.v3 as iio
from tqdm import tqdm


def calculate_hue_similarity(img1, img2):
    # 将图像从BGR转换到HSV颜色空间
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # 提取色调分量
    img1_hue = img1_hsv[:, :, 0]
    img2_hue = img2_hsv[:, :, 0]

    # 将色调值归一化到[0, 1]区间
    img1_hue = img1_hue / 360.0
    img2_hue = img2_hue / 360.0

    # 计算两个色调数组的均值
    mean_hue1 = np.mean(img1_hue)
    mean_hue2 = np.mean(img2_hue)

    # 计算余弦相似度
    similarity = np.sum(img1_hue * img2_hue) / (np.sqrt(np.sum(img1_hue ** 2)) * np.sqrt(np.sum(img2_hue ** 2)))

    return similarity


gt_list = []
pred_list = []
for i  in tqdm(range(1200)):
    gif = iio.imread(os.path.join('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub01/recons_and_gt/', f'test_and_gt{i+1}.gif'), index=None)
    gt, pred = np.split(gif, 2, axis=2)
    #print(gt)
    gt_list.append(gt)
    pred_list.append(pred)

gt_list = np.stack(gt_list)
pred_list = np.stack(pred_list)
print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')


C = []
for i in tqdm(range(gt_list.shape[0])):
    corr = []
    for j in range(6):
        recons = pred_list[i,j,:,:,:]
        gt = gt_list[i,j,:,:,:]
        c = calculate_hue_similarity(recons, gt)
        corr.append(c)
    s = np.mean(corr)
    C.append(s)

np.save('/data0/home/luyizhuo/NIPS2024实验材料/消融实验/模块消融/假设检验数据存放/huePCC_full.npy', np.array(C))
print(np.mean(C))
print(np.std(C))
