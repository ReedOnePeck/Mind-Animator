from skimage.metrics import structural_similarity as SSIM
import torch
import imageio
from tqdm import tqdm
from einops import rearrange
import numpy as np
import os
import imageio.v3 as iio
import math
import cv2
import re
import matplotlib.pyplot as plt

def compute_epe(flow_gt, flow_pred):
    epe = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    return epe


def compute_ae(flow_gt, flow_pred):
    dot_product = np.sum(flow_gt * flow_pred, axis=-1)
    norm_gt = np.linalg.norm(flow_gt, axis=-1)
    norm_pred = np.linalg.norm(flow_pred, axis=-1)
    cos_theta = dot_product / (norm_gt * norm_pred + 1e-8)
    ae = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    ae = np.degrees(ae)
    return ae


def compute_optical_flow(frames):
    optical_flows = []
    for i in range(frames.shape[0] - 1):
        prev_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flows.append(flow)
    return optical_flows


def calculate_epe_ae(gt_ndarray, recons_ndarray):
    mean_epe_list = []
    mean_ae_list = []

    # Iterate through each video
    for i in tqdm(range(gt_ndarray.shape[0])):
        gif_frames = [frame for frame in recons_ndarray[i]]
        gt_resized = [frame for frame in gt_ndarray[i]]


        pred_flows = compute_optical_flow(np.array(gif_frames))
        gt_flows = compute_optical_flow(np.array(gt_resized))


        # Calculate EPE and AE
        epe = compute_epe(np.array(gt_flows), np.array(pred_flows))
        ae = compute_ae(np.array(gt_flows), np.array(pred_flows))

        # Store the mean EPE and AE for this video
        mean_epe_list.append(np.mean(epe))
        mean_ae_list.append(np.mean(ae))

    # Calculate the overall mean EPE and AE
    overall_mean_epe = np.mean(mean_epe_list)
    overall_mean_ae = np.mean(mean_ae_list)

    return mean_epe_list, overall_mean_ae



gt_list = []

#[22,43,47,74,75,145,191,227,228,230,249,260,262,333,335,452,453,470,487,493,509,579,661,677,682,712,733,782,786,804,812,813,842,874,880,889,936,961,1008,1009,1019,1074,1152,1170]
for k  in tqdm([22,43,47,74,75,145,191,227,228,230,249,260,262,333,335,452,453,470,487,493,509,579,661,677,682,712,733,782,786,804,812,813,842,874,880,889,936,961,1008,1009,1019,1074,1152,1170]):
    gif = iio.imread(os.path.join('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub01/recons_and_gt/', f'test_and_gt{k}.gif'), index=None)
    gt, _ = np.split(gif, 2, axis=2)
    #print(gt)
    gt_list.append(np.array([cv2.resize(gt_f, (512, 512)) for gt_f in gt]))
gt_list = np.stack(gt_list)

print(f'gt shape: {gt_list.shape}')

pred_list = []
for i  in tqdm([22,43,47,74,75,145,191,227,228,230,249,260,262,333,335,452,453,470,487,493,509,579,661,677,682,712,733,782,786,804,812,813,842,874,880,889,936,961,1008,1009,1019,1074,1152,1170]):
    f_recons = []
    with imageio.get_reader(
            '/data0/home/luyizhuo/NIPS2024实验材料/消融实验/模块消融/去掉z_consist/重建结果/test{}.gif'.format(i)) as gif:
        for j, frame in enumerate(gif):
            f = np.array(frame)
            f_recons.append(f)
    pred_list.append(np.array(f_recons))
pred_list = np.stack(pred_list)


overall_mean_epe, overall_mean_ae = calculate_epe_ae(gt_list, pred_list)
print("Overall Mean EPE:", np.mean(overall_mean_epe))
print("Overall std EPE:", np.std(overall_mean_epe))

data = overall_mean_epe
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # 设置直方图的外观
plt.title("Distribution of Data", fontsize=16)  # 设置标题
plt.xlabel("Value", fontsize=14)  # 设置X轴标签
plt.ylabel("Frequency", fontsize=14)  # 设置Y轴标签
plt.grid(axis='y', alpha=0.75)  # 添加网格线，仅作用于y轴
plt.show()



P = []
for i in range(5):
    EPE_mean = []
    EPE_K  = 0


    for num in tqdm(range(100)):
        shuffled_indices = np.random.permutation(8)
        Pred_list = pred_list[:, shuffled_indices, :, :, :]
        shuffle_mean_epe, _ = calculate_epe_ae(gt_list, Pred_list)
        EPE_mean.append(np.mean(shuffle_mean_epe))
        print(np.mean(shuffle_mean_epe))


    for j in range(100):
        if EPE_mean[j] > np.mean(overall_mean_epe):
            EPE_K  = EPE_K  + 1
            print("+1")
        else:
            print("+0")

    print(100-EPE_K)

    #sub1
    #100-23
    P.append(100 - EPE_K)

print(P)



















