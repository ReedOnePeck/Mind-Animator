#该代码需要在BLIP2环境中运行,
import torch
import os
import random
from tqdm import tqdm
import h5py
import numpy as np
dtype = h5py.special_dtype(vlen=str)
from PIL import Image
from lavis.models import load_model_and_preprocess
device = torch.device('cuda:5')
model,vis_processors,_=load_model_and_preprocess(name="blip2_t5",model_type="pretrain_flant5xxl", is_eval=True, device=device)

video_path_root = '/data0/home/luyizhuo/HCP预处理/video_frames/'
Captions = []
for idx in tqdm(range(3040)):
    frames_root = video_path_root + 'video{}_frames/'.format(idx+1)
    frame1 = Image.open(frames_root + '0000012.jpg').convert("RGB")
    frame1 = vis_processors["eval"](frame1).unsqueeze(0).to(device)

    caption1 = model.generate({"image": frame1, "prompt": "Question: What does this image describe? Answer:"})
    caption = caption1[0]
    Captions.append(caption)

Captions = np.array(Captions , dtype=dtype)
f = h5py.File('/data0/home/luyizhuo/HCP预处理/Video_Captions.h5', 'w')
f['Captions'] = Captions
f.close()


