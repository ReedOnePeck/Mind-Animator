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
from typing import Callable, List, Optional, Union
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from PIL import Image
import clip


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
for i in tqdm(range(1200)):
    gif = iio.imread(os.path.join(
        '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub01/recons_and_gt/',
        f'test_and_gt{i + 1}.gif'), index=None)
    gt, pred = np.split(gif, 2, axis=2)
    gt_list.append(np.array([cv2.resize(gt_f, (512, 512)) for gt_f in gt]))

gt_list = np.stack(gt_list)

print(f'gt shape: {gt_list.shape}')


for sub_ID in range(2,4):
    print('================================sub{}====================================='.format(sub_ID))


    pred_list = []
    for i in tqdm(range(1200)):
        f_recons = []
        with imageio.get_reader(
                '/data0/home/luyizhuo/recons/sub{}/recons/test{}.gif'.format(sub_ID, i + 1)) as gif:
            for j, frame in enumerate(gif):
                f = np.array(frame)
                f_recons.append(f)
        pred_list.append(np.array(f_recons))
    pred_list = np.stack(pred_list)
    print(f'pred_list shape: {pred_list.shape}')


    overall_mean_epe, overall_mean_ae = calculate_epe_ae(gt_list, pred_list)
    print("Overall Mean EPE:", np.mean(overall_mean_epe))
    np.save(f'/data0/home/luyizhuo/假设检验结果存放/sub{sub_ID}/EPE.npy',
            np.array(overall_mean_epe))