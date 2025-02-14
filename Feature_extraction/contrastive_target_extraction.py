import torch
import clip
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
device = torch.device('cuda:5')

weight_dtype = torch.float32
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(torch.float32)

#Text----------------------------------------------------------------------------------------------------------------------------------------------------

Train_prompts = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_captions_new.h5', 'r')
Train_captions = Train_prompts['Train_captions']
Train_embeddings = []
for j in tqdm(range(len(Train_captions))):
    caption = str(Train_captions[j],'utf-8')
    text = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().detach().numpy()
    Train_embeddings.append(text_features)
Train_ = np.concatenate(Train_embeddings,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_text_embeddings_CLIP_512.npy',Train_)

Test_prompts = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_captions_new.h5', 'r')
Test_captions = Test_prompts['Test_captions']
Test_embeddings = []
for j in tqdm(range(len(Test_captions))):
    caption = str(Test_captions[j],'utf-8')
    text = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().detach().numpy()
    Test_embeddings.append(text_features)
Test_ = np.concatenate(Test_embeddings,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_text_embeddings_CLIP_512.npy',Test_)

#Img----------------------------------------------------------------------------------------------------------------------------------------------------
Train_video_path_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/'
Train_imgs = []
for i in tqdm(range(18)):
    for j in tqdm(range(240)):
        frames_root = Train_video_path_root + 'seg{}_{}/'.format(i + 1, j + 1)
        image = preprocess(Image.open(frames_root + '0000024.jpg')).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).cpu().detach().numpy()
        Train_imgs.append(image_features)
Train_ = np.concatenate(Train_imgs,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_img_embeddings_CLIP_512.npy',Train_)


Test_video_path_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/'
Test_imgs = []
for i in tqdm(range(5)):
    for j in tqdm(range(240)):
        frames_root = Test_video_path_root + 'test{}_{}/'.format(i + 1, j + 1)
        image = preprocess(Image.open(frames_root + '0000024.jpg')).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).cpu().detach().numpy()
        Test_imgs.append(image_features)
Test_ = np.concatenate(Test_imgs,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_img_embeddings_CLIP_512.npy',Test_)



