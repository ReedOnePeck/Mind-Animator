import numpy as np
from transformers import pipeline
from typing import Callable, List, Optional, Union
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from einops import rearrange
import imageio.v3 as iio
from tqdm import tqdm
import imageio
import cv2
device = torch.device('cuda:1')


def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    if isinstance(class_id, int):
        class_id = [class_id]
    pick_range = [i for i in np.arange(len(pred)) if i not in class_id]
    corrects = 0
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=False)
        for gt_id in class_id:
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]
            if 0 in pred_picked:
                corrects += 1
                break
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)

processor = ViTImageProcessor.from_pretrained('/data0/home/luyizhuo/Pretrained_Models/vit-base-patch16-224',
                                                  cache_dir='.cache')
model = ViTForImageClassification.from_pretrained('/data0/home/luyizhuo/Pretrained_Models/vit-base-patch16-224',
                                                  cache_dir='.cache').to(device, torch.float16)
model.eval()

print('模型加载完毕')

@torch.no_grad()
def img_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: int = 50,
        num_trials: int = 100,
        top_k: int = 1,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False
):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    assert n_way > top_k


    acc_list = []
    std_list = []
    for pred, gt in zip(pred_videos, gt_videos):
        pred = processor(images=pred.astype(np.uint8), return_tensors='pt')
        gt = processor(images=gt.astype(np.uint8), return_tensors='pt')
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1, descending=False).detach().flatten()[-3:]
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)
    if return_std:
        return acc_list, std_list
    return acc_list

gt_video_sub1 = rearrange(torch.tensor(np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/test_gt_npy/test_groundtruth.npy')), 'b c f h w -> b f h w c').numpy()

gt_list_512 = []
pred_list_512 = []
for i  in tqdm(range(1200)):
    f_gt = gt_video_sub1[i, :, :, :, :] * 255.  # img:0~255
    gt_list_512.append(f_gt)

    f_recons = []
    with imageio.get_reader(
            '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/sub02/recons_only/test{}.gif'.format(
                i + 1)) as gif:
        for j, frame in enumerate(gif):
            f = np.array(frame)
            f_recons.append(f)

    pred_list_512.append(np.array(f_recons))


gt_list = np.stack(gt_list_512)
pred_list = np.stack(pred_list_512)
print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')




n_way = 2
num_trials = 20
top_k = 1

mean = []
std = []

for j in range(8):
    recons = pred_list[:,j,:,:,:]
    gt = gt_list[:,j,:,:,:]
    acc_list, std_list = img_classify_metric(
                            recons,
                            gt,
                            n_way=n_way,
                            top_k=top_k,
                            num_trials=num_trials,
                            return_std=True,
                            device=device)
    mean.append(np.mean(acc_list))
    std.append(np.mean(std_list))

print('2way 的准确率均值和标准差为：')
print(np.mean(mean))
print(np.mean(std))



