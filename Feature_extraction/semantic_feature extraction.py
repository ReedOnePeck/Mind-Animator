import numpy as np
import h5py
import torch
import decord
decord.bridge.set_bridge('torch')
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:0')
seed = 1240


weight_dtype = torch.float64
tokenizer = CLIPTokenizer.from_pretrained("/data0/home/luyizhuo/Tune-a-video_checkpoints/tokenizer")
text_encoder = CLIPTextModel.from_pretrained("/data0/home/luyizhuo/Tune-a-video_checkpoints/text_encoder").to(device, dtype=weight_dtype)
text_encoder.requires_grad_(False)


def _encode_prompt( prompt, device, num_videos_per_prompt):

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=20  , #self.tokenizer.model_max_length
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    return text_embeddings


Train_prompts = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_captions_new.h5', 'r')
Train_captions = Train_prompts['Train_captions']
Train_embeddings = []
for j in tqdm(range(len(Train_captions))):
    caption = str(Train_captions[j],'utf-8')
    #caption = ['hello', 'word']
    embedding = _encode_prompt(prompt=caption, device=device, num_videos_per_prompt=1).cpu().detach().numpy()
    Train_embeddings.append(embedding)
Train_ = np.concatenate(Train_embeddings,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_embeddings_new.npy',Train_)

Test_prompts = h5py.File('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_captions_new.h5', 'r')
Test_captions = Test_prompts['Test_captions']
Test_embeddings = []
for j in tqdm(range(len(Test_captions))):
    caption = str(Test_captions[j],'utf-8')
    embedding = _encode_prompt(prompt=caption, device=device, num_videos_per_prompt=1).cpu().detach().numpy()
    Test_embeddings.append(embedding)
Test_ = np.concatenate(Test_embeddings,axis = 0)
np.save('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/Test_embeddings_new.npy',Test_)


