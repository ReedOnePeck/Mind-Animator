import numpy as np
from typing import Callable, List, Optional, Union
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import os
from einops import rearrange
import imageio
import cv2
import imageio.v3 as iio
from tqdm import tqdm
device = torch.device('cuda:5')


def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    if isinstance(class_id, int):
        class_id = [class_id]
    pick_range =[i for i in np.arange(len(pred)) if i not in class_id]
    corrects = 0
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        for gt_id in class_id:
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]
            if 0 in pred_picked:
                corrects += 1
                break
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)


@torch.no_grad()
def video_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: int = 50,
        num_trials: int = 100,
        top_k: int = 1,
        num_frames: int = 6,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False
):
    # pred_videos: n, 6, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 6, 256, 256, 3 in pixel values: 0 ~ 255
    assert n_way > top_k
    processor = VideoMAEImageProcessor.from_pretrained('/data0/home/luyizhuo/Pretrained_Models/VideoMAE',
                                                       cache_dir=cache_dir)
    model = VideoMAEForVideoClassification.from_pretrained('/data0/home/luyizhuo/Pretrained_Models/VideoMAE',
                                                           num_frames=num_frames,
                                                           cache_dir=cache_dir).to(device, torch.float16)
    model.eval()

    acc_list = []
    std_list = []

    for pred, gt in tqdm(zip(pred_videos, gt_videos)):
        pred = processor(list(pred), return_tensors='pt')
        gt = processor(list(gt), return_tensors='pt')
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1, descending=False).detach().flatten()[-3:]
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)
    if return_std:
        return acc_list, std_list
    return acc_list


gt_list = []
recons_list = []

for i  in tqdm(range(1200)):
    gif = iio.imread(os.path.join('/data0/home/luyizhuo/NIPS2024实验材料/sub3/sample-all/', f'test{i+1}.gif'), index=None)
    gt,recons = np.split(gif, 2, axis=2)
    gt_list.append(np.array([cv2.resize(gt_f, (512, 512)) for gt_f in gt]))
    recons_list.append(np.array([cv2.resize(recons_f, (512, 512)) for recons_f in recons]))
gt_list = np.stack(gt_list)
recons_list = np.stack(recons_list)
print(f'gt shape: {gt_list.shape}')





n_way = 2
num_trials = 100
top_k = 1
# video classification scores
acc_list, std_list = video_classify_metric(
                                    recons_list,
                                    gt_list,
                                    n_way = n_way,
                                    top_k=top_k,
                                    num_trials=num_trials,
                                    num_frames=gt_list.shape[1],
                                    return_std=True,
                                    device=device
                                    )

print(f'2Way video classification score: {np.mean(acc_list)} +- {np.mean(std_list)}')