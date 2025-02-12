#该代码需要在BLIP2环境中运行,
import torch
import os
import random
from tqdm import tqdm
import h5py
import numpy as np
dtype = h5py.special_dtype(vlen=str)
from PIL import Image
import clip
from lavis.models import load_model_and_preprocess
device = torch.device('cuda:5')
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
model,vis_processors,_=load_model_and_preprocess(name="blip2_t5",model_type="pretrain_flant5xxl", is_eval=True, device=device)

#保存训练集video标签-----------------------------------------------------------------------------------------------------------------------------
Train_video_path_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/'
Train_Captions = []
for i in tqdm(range(18)):
    for j in tqdm(range(240)):
        frames_root = Train_video_path_root + 'seg{}_{}/'.format(i + 1, j + 1)
        frame1 = Image.open(frames_root + '0000000.jpg').convert("RGB") ;frame1 = vis_processors["eval"](frame1).unsqueeze(0).to(device)
        frame2 = Image.open(frames_root + '0000056.jpg').convert("RGB") ;frame2 = vis_processors["eval"](frame2).unsqueeze(0).to(device)
        frame_mid = Image.open(frames_root + '0000024.jpg').convert("RGB") ;clip_image = clip_preprocess(frame_mid).unsqueeze(0).to(device)

        caption1 = model.generate({"image": frame1, "prompt": "Question: What does this image describe? Answer:"})
        caption2 = model.generate({"image": frame2, "prompt": "Question: What does this image describe? Answer:"})

        text1 = clip.tokenize(caption1).to(device)
        text2 = clip.tokenize(caption2).to(device)

        with torch.no_grad():
            image_features3 = clip_model.encode_image(clip_image)
            text_features1 = clip_model.encode_text(text1)
            text_features2 = clip_model.encode_text(text2)
            cos_sim1 = torch.cosine_similarity(image_features3, text_features1).cpu().detach().numpy()
            cos_sim2 = torch.cosine_similarity(image_features3, text_features2).cpu().detach().numpy()
            # 0.2477  0.2686  0.3074  0.2827  0.1835
            # 0.2864  0.1838  0.3188  0.2742  0.2698
        if abs(cos_sim1-cos_sim2)<=0.05:
            number = random.random()
            if number>=0.5:
                caption = caption1[0]
            else:
                caption = caption2[0]
        else:
            caption = caption1[0] + ', and then ' +caption2[0]
        Train_Captions.append(caption)
Train_Captions = np.array(Train_Captions , dtype=dtype)
f_Train = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_captions_new.h5', 'w')
f_Train['Train_captions'] = Train_Captions
f_Train.close()

#保存测试集video标签-----------------------------------------------------------------------------------------------------------------------------
Test_video_path_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/'
Test_Captions = []
for i in tqdm(range(5)):
    for j in tqdm(range(240)):
        frames_root = Test_video_path_root + 'test{}_{}/'.format(i + 1, j + 1)
        frame1 = Image.open(frames_root + '0000000.jpg').convert("RGB") ;frame1 = vis_processors["eval"](frame1).unsqueeze(0).to(device)
        frame2 = Image.open(frames_root + '0000056.jpg').convert("RGB") ;frame2 = vis_processors["eval"](frame2).unsqueeze(0).to(device)
        frame_mid = Image.open(frames_root + '0000024.jpg').convert("RGB") ;clip_image = clip_preprocess(frame_mid).unsqueeze(0).to(device)

        caption1 = model.generate({"image": frame1, "prompt": "Question: What does this image describe? Answer:"})
        caption2 = model.generate({"image": frame2, "prompt": "Question: What does this image describe? Answer:"})

        text1 = clip.tokenize(caption1).to(device)
        text2 = clip.tokenize(caption2).to(device)

        with torch.no_grad():
            image_features3 = clip_model.encode_image(clip_image)
            text_features1 = clip_model.encode_text(text1)
            text_features2 = clip_model.encode_text(text2)
            cos_sim1 = torch.cosine_similarity(image_features3, text_features1).cpu().detach().numpy()
            cos_sim2 = torch.cosine_similarity(image_features3, text_features2).cpu().detach().numpy()
        if abs(cos_sim1 - cos_sim2) <= 0.05:
            number = random.random()
            if number >= 0.5:
                caption = caption1[0]
            else:
                caption = caption2[0]
        else:
            caption = caption1[0] + ', and then ' + caption2[0]
        Test_Captions.append(caption)
Test_Captions = np.array(Test_Captions, dtype=dtype)
f_Test = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_captions_new.h5','w')
f_Test['Test_captions'] = Test_Captions
f_Test.close()

