import torch
import clip
from PIL import Image
import os
from einops import rearrange
import imageio
import cv2
import imageio.v3 as iio
from tqdm import tqdm
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda:0')
model, preprocess = clip.load("ViT-B/32", device=device)

def cal_CLIP_pcc( predlist, VIFI_CLIP_list, bar):
    pcc_list = []
    for i in range(predlist.shape[0]):
        if VIFI_CLIP_list[i] <= bar:
            pcc_list.append(0)
        else:
            cos_simlist = []
            pred_frames = predlist[i,:,:,:,:]
            for j in range(pred_frames.shape[0]-1):
                img_recons1 = Image.fromarray(np.uint8(pred_frames[j, :, :, :]))
                recons1 = preprocess(img_recons1).unsqueeze(0).to(device)
                recons1_features = model.encode_image(recons1)

                img_recons2 = Image.fromarray(np.uint8(pred_frames[j + 1, :, :, :]))
                recons2 = preprocess(img_recons2).unsqueeze(0).to(device)
                recons2_features = model.encode_image(recons2)

                cos_sim = torch.cosine_similarity(recons1_features, recons2_features).cpu().detach().numpy()
                cos_simlist.append(cos_sim)
            pcc_list.append(np.mean(cos_simlist))
    return pcc_list


pred_list_sub2 = []
for i  in tqdm(range(1200)):
    f_recons = []
    with imageio.get_reader(
            '/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub02/reconstructions/test{}.gif'.format(i + 1)) as gif:
        for j, frame in enumerate(gif):
            f = np.array(frame)
            f_recons.append(f)
    pred_list_sub2.append(np.array(f_recons))
pred_list_sub2 = np.stack(pred_list_sub2)
VIFI_CLIP_sub2 = np.load('/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub02/假设检验结果存放/VIFICLIP_2.npy').flatten()

c = cal_CLIP_pcc(pred_list_sub2, VIFI_CLIP_sub2, 0.6)
print("pred_list_sub2的CLIP_consist计算完成,bar = 0.6")
print(np.mean(c))
print(np.std(c))
np.save('/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub02/假设检验结果存放/consist_pcc_pred_list_sub2.npy',np.array(c))
