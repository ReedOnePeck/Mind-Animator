import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
import decord
decord.bridge.set_bridge('torch')
from torch.autograd import Variable
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)

from einops import rearrange

from MindAnimator.models.Inflated_SD_by_Wu.unet import UNet3DConditionModel
from MindAnimator.models.utils import save_videos_grid, ddim_inversion
from MindAnimator.models.Inflated_SD_pipeline import Inflated_SD_Pipeline
from MindAnimator.models.Decoder import make_CMG_model, Semantic_decoder, Strcuture_decoder, subsequent_mask

device = torch.device('cuda:2')


def greedy_decode(first_frame,  model, src, src_mask,  max_len ,mask_ratio):
    decoder_input = first_frame
    src = torch.tensor(src)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(1).type_as(src.data)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(decoder_input),
                           Variable(subsequent_mask(ys.size(1), mask_ratio).type_as(src.data).to(device))   )
        next_frame = (out[:,-1,:]).unsqueeze(dim=1)
        ys = torch.cat([ys,torch.ones(1, 1).fill_(1).type_as(src.data)], dim=1)
        decoder_input = torch.cat([decoder_input,next_frame] , dim=1)

    return decoder_input


weight_dtype = torch.float32
noise_scheduler = DDPMScheduler.from_pretrained("/data0/home/luyizhuo/Stable_diffusion_ckpt/scheduler")

inference_scheduler = DDIMScheduler.from_pretrained("/data0/home/luyizhuo/Stable_diffusion_ckpt/scheduler")
tokenizer = CLIPTokenizer.from_pretrained("/data0/home/luyizhuo/Stable_diffusion_ckpt/tokenizer")
text_encoder = CLIPTextModel.from_pretrained("/data0/home/luyizhuo/Stable_diffusion_ckpt/text_encoder").to(device, dtype=weight_dtype)
vae = AutoencoderKL.from_pretrained("/data0/home/luyizhuo/Stable_diffusion_ckpt/vae").to(device, dtype=weight_dtype)
unet = UNet3DConditionModel.from_pretrained_2d("/data0/home/luyizhuo/Stable_diffusion_ckpt/unet").to(device, dtype=weight_dtype)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
unet.enable_xformers_memory_efficient_attention()

inference_pipeline = Inflated_SD_Pipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,scheduler=inference_scheduler)
inference_pipeline.enable_vae_slicing()
ddim_inv_scheduler = inference_scheduler
ddim_inv_scheduler.set_timesteps(50)

generator = torch.Generator(device=device)




def main():
    parser = argparse.ArgumentParser(description='video recons')
    parser.add_argument('--semantic_model_path', help='model saved path', default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/semantic/Semantic_decoder_30.pth', type=str)
    parser.add_argument('--structure_model_path', help='model saved path',default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/structure/Structure_decoder_75.pth', type=str)
    parser.add_argument('--CMG_model_path', help='model saved path',default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/CMG/CMG_patch_size_64_epoch70.pth', type=str)
    parser.add_argument('--video_save_folder1', help='model saved path',default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/results_recons/', type=str)
    parser.add_argument('--video_save_folder2', help='model saved path',default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/results_recons_with_gt/', type=str)
    parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
    parser.add_argument('--random_seed', help='random_seed', default=42, type=int)

    args = parser.parse_args()
    subj_ID = args.subj_ID
    seed = args.random_seed
    generator.manual_seed(seed)

    model = Semantic_decoder(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False)
    model.load_state_dict(torch.load(args.semantic_model_path.foramt(subj_ID)))
    model.to(device)
    model.eval()

    model_first = Strcuture_decoder(in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=2, norm_type='ln', act_first=False)
    model_first.load_state_dict(torch.load(args.structure_model_path.foramt(subj_ID)))
    model_first.to(device)
    model_first.eval()

    model_content = make_CMG_model(2, d_model=4096*4, d_ff=768, h=4, dropout=0.2)
    model_content.load_state_dict(torch.load(args.CMG_model_path.foramt(subj_ID)))
    model_content.to(device)
    model_content.eval()


    test_src_path = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Test/activated_voxels_PCA/masked4500_test_data.npy'.format(subj_ID)
    test_src = np.load(test_src_path)

    test_gts = torch.tensor(np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/reconstruction_results/test_gt_npy/test_groundtruth.npy'), dtype=torch.float32)
    for video_ID in tqdm(range(1200)):
        fMRI = torch.tensor(test_src[video_ID:video_ID + 1, :], dtype=torch.float32).to(device)

        first_frame = model_first(fMRI).to(torch.float32)
        latents = rearrange(greedy_decode(first_frame=first_frame, model=model_content, src=fMRI, src_mask=None, max_len=8, mask_ratio=0), "b f (c h w) -> b c f h w ", f=8, c=4, h=64, w=64)#.view(-1,4,8,32,32)
        ddim_inv_latent = ddim_inversion(inference_pipeline, ddim_inv_scheduler, video_latent=latents, num_inv_steps=50, prompt="")[-1].to(weight_dtype)

        noise = torch.randn_like(latents)
        timesteps = torch.tensor([200]).long()
        noisy_latents = noise_scheduler.add_noise(ddim_inv_latent, noise, timesteps)
        cls = np.expand_dims(np.load('/nfs/nica-datashop/CC2017_for_video_reconstruction/semantic_decoder/cls.npy'),0)
        _, sem = model(fMRI)
        sem = sem.cpu().detach().numpy()
        prompt = torch.tensor(np.concatenate([cls, sem], axis=1), dtype=torch.float32).to(device)

        # --------------------------------------------------------------------------------------------------------------------------------------
        output_dir1 = args.video_save_folder1.foramt(subj_ID)
        output_dir2 = args.video_save_folder2.foramt(subj_ID)

        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
            os.makedirs(output_dir2)

        sample = inference_pipeline(prompt, generator=generator, latents=noisy_latents).videos
        gt = test_gts[video_ID:video_ID + 1,:,:,:,:]

        save_videos_grid(sample, output_dir1 + '/test{}.gif'.format(video_ID+1))
        save_videos_grid(torch.concat([gt,sample], dim=0), output_dir2 + '/test_and_gt{}.gif'.format(video_ID+1))


