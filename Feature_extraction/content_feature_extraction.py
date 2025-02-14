import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)
from tqdm import tqdm
from einops import rearrange
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda:0')
seed = 1023


weight_dtype = torch.float32
noise_scheduler = DDPMScheduler.from_pretrained("/data0/home/luyizhuo/Tune-a-video_checkpoints/scheduler")
inference_scheduler = DDIMScheduler.from_pretrained("/data0/home/luyizhuo/Tune-a-video_checkpoints/scheduler")
vae = AutoencoderKL.from_pretrained("/data0/home/luyizhuo/Tune-a-video_checkpoints/vae").to(device, dtype=weight_dtype)
vae.requires_grad_(False)

video_path = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_clips/{}/'

file_names_Train = []
for i in range(1,19):
    for j in range(1,241):
        file_names_Train.append('seg{}_{}.mp4'.format(i,j))

file_names_Test = []
for i in range(1,6):
    for j in range(1,241):
        file_names_Test.append('test{}_{}.mp4'.format(i,j))


@torch.no_grad()
def feature_extractor(data_clas):
    content_latents = []
    path = video_path.format(data_clas)
    if data_clas=='Train':
        file_names = file_names_Train
    else:
        file_names = file_names_Test

    for j in tqdm(range(len(file_names))):
        file_name = file_names[j]
        video_file_path = path + file_name
        vr = decord.VideoReader(video_file_path, width=256,height=256)
        sample_index = np.array(list(range(0, len(vr), 1)))[[0,8,16,24,32,40,48,56]]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        video = video.unsqueeze(0)

        pixel_values = (video / 127.5 - 1.0).to(device, weight_dtype)
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = (latents * 0.18215).cpu().numpy()
        content_latents.append(latents)

    all_contents = np.concatenate(content_latents,axis = 0)
    np.save(path + 'contents256_float32.npy',all_contents)
    print(all_contents.shape)
    return

a = feature_extractor(data_clas = 'Train')
b = feature_extractor(data_clas = 'Test')