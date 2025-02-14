import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import h5py
import clip
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import lr_scheduler
from MindAnimator.models.data_aug import eda,fMRI_sparsification
from MindAnimator.models.Decoder import Semantic_decoder
device = torch.device('cuda:2')

# ----------------------------------------------------------Model------------------------------------------------------------------
def make_model(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False):
    model = Semantic_decoder(in_dim=in_dim, out_dim=out_dim, h=h, n_blocks=n_blocks, norm_type=norm_type,
                         act_first=act_first)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# ----------------------------------------------------------Data preparation------------------------------------------------------------------
transform = transforms.Compose([
    #transforms.ToTensor(),
    transforms.RandomCrop(size=(400, 400)),
    transforms.Resize((224, 224)),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

model, _ = clip.load("ViT-B/32", device=device)
model = model.to(torch.float32)

weight_dtype = torch.float32
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




class Train_dataset(Dataset):
    def __init__(self, f_MRI_data, image_data, trn_tgt_text):
        #fMRI
        self.trn_src = torch.tensor(fMRI_sparsification(data=f_MRI_data[:4000,:],rate1=0.2 , rate2=0.5), dtype=torch.float32).to(device)
        #Img
        trn_tgt_picture = transform(image_data[:4000,:,:,:])
        with torch.no_grad():
            self.trn_tgt_CLIP_picture = model.encode_image(trn_tgt_picture)

        #Text
        captions = []
        for j in range(4000):
            text = eda( str(trn_tgt_text[j], 'utf-8') )
            captions.append(text)
        with torch.no_grad():
            tokenized = clip.tokenize(captions).to(device)
            self.trn_tgt_CLIP_text = model.encode_text(tokenized)
            self.embedding = _encode_prompt(prompt=captions, device=device, num_videos_per_prompt=1)

    def __len__(self):
        data_size = 4000
        return data_size

    def __getitem__(self, item):
        trn_src = self.trn_src[item, :]

        text_embedding = self.embedding[item, 1:, :]

        trn_tgt_CLIP_picture = self.trn_tgt_CLIP_picture[item, :]

        trn_tgt_CLIP_text = self.trn_tgt_CLIP_text[item, :]
        return trn_src, text_embedding, trn_tgt_CLIP_text, trn_tgt_CLIP_picture

class Val_dataset(Dataset):
    def __init__(self, f_MRI_data, image_data, trn_tgt_text):
        # fMRI
        self.trn_src = torch.tensor(f_MRI_data[4000:, :],dtype=torch.float32).to(device)
        # Img
        trn_tgt_picture = transform(image_data[4000:, :, :, :])
        with torch.no_grad():
            self.trn_tgt_CLIP_picture = model.encode_image(trn_tgt_picture)

        # Text
        captions = []
        for j in range(4000,4320):
            text = str(trn_tgt_text[j], 'utf-8')
            captions.append(text)
        with torch.no_grad():
            tokenized = clip.tokenize(captions).to(device)
            self.trn_tgt_CLIP_text = model.encode_text(tokenized)
            self.embedding = _encode_prompt(prompt=captions, device=device, num_videos_per_prompt=1)

    def __len__(self):
        data_size = 320
        return data_size

    def __getitem__(self, item):
        trn_src = self.trn_src[item, :]

        text_embedding = self.embedding[item, 1:, :]

        trn_tgt_CLIP_picture = self.trn_tgt_CLIP_picture[item, :]

        trn_tgt_CLIP_text = self.trn_tgt_CLIP_text[item, :]
        return trn_src, text_embedding, trn_tgt_CLIP_text, trn_tgt_CLIP_picture

# ----------------------------------------------------------Train------------------------------------------------------------------
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.01

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))




def contrastive_loss(logits):
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity):
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class Contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_per_unit):
        loss = clip_loss(logits_per_unit)
        return loss


class SimpleLossCompute:
    def __init__(self, criterion1, criterion2, k1, k2,  opt=None):
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.k1 = k1
        self.k2 = k2
        self.opt = opt

    def __call__(self, mode, epoch=None,y_pred=None, regularization=None, y_pred_mid=None, logits_per_fMRI_t=None,
                 logits_per_fMRI_i=None, y_tgt=None, y_CLIP_t=None, y_CLIP_i=None, norm=None):
        if mode == 'train':
            loss_pre = self.criterion2(y_pred_mid, y_CLIP_t) / norm

            #loss_reg = regularization
            loss_fMRI_text = self.criterion1(logits_per_fMRI_t) / norm
            loss_fMRI_img = self.criterion1(logits_per_fMRI_i) / norm
            loss_proj = self.criterion2(y_pred, y_tgt) / norm
            loss_contrast = 0.5 * (loss_fMRI_text + loss_fMRI_img)

            loss = loss_pre + self.k1 * loss_contrast + self.k2 * loss_proj

            return loss

        if mode == 'val':
            loss_pre = self.criterion2(y_pred_mid, y_CLIP_t) / norm

            # loss_reg = regularization
            loss_fMRI_text = self.criterion1(logits_per_fMRI_t) / norm
            loss_fMRI_img = self.criterion1(logits_per_fMRI_i) / norm
            loss_proj = self.criterion2(y_pred, y_tgt) / norm
            loss_contrast = 0.5 * (loss_fMRI_text + loss_fMRI_img)

            loss = loss_pre + self.k1 * loss_contrast + self.k2 * loss_proj

            return  loss.data.item() * norm


def run_epoch(mode, data_iter, model, loss_compute, norm=1, epoch=None, opt=None, scheduler=None):
    val_loss = []
    train_loss = []
    total_loss = 0
    for i, batch in tqdm(enumerate(data_iter)):
        if mode == 'train':
            loss_reg = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    loss_reg += torch.sum(torch.abs(param))

            trn_src, trn_proj, trn_tgt_CLIP_text, trn_tgt_CLIP_picture = batch
            x, y, logits_per_fMRI_t, logits_per_fMRI_i = model.forward(x=trn_src,
                                                                       y_CLIP_t=trn_tgt_CLIP_text,
                                                                       y_CLIP_i=trn_tgt_CLIP_picture)
            loss = loss_compute(mode=mode, epoch=epoch, y_pred_mid=x,y_CLIP_t=trn_tgt_CLIP_text,y_pred=y, y_tgt=trn_proj,
                                logits_per_fMRI_t=logits_per_fMRI_t,logits_per_fMRI_i=logits_per_fMRI_i, norm=norm)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.data.item()
            train_loss.append(loss.data.item() )
            if i % 10 == 0:
                print('当前状态为{},进行到第{}个batch，这一个batch的总损失函数为{}'.format(mode, i, loss.data.item()))

        if mode == 'val':
            val_src, val_proj, val_tgt_CLIP_text, val_tgt_CLIP_picture = batch
            x, y, logits_per_fMRI_t, logits_per_fMRI_i = model.forward(x=val_src,
                                                                       y_CLIP_t=val_tgt_CLIP_text,
                                                                       y_CLIP_i=val_tgt_CLIP_picture)
            t_loss = loss_compute(mode=mode, epoch=epoch, y_pred_mid=x,y_CLIP_t=val_tgt_CLIP_text,y_pred=y, y_tgt=val_proj,
                                logits_per_fMRI_t=logits_per_fMRI_t,logits_per_fMRI_i=logits_per_fMRI_i, norm=norm)
            #pcc.append(cor)
            val_loss.append(t_loss)
            if i % 10 == 0:
                print('当前状态为{},进行到第{}个batch，这一个batch的损失函数为{}'.format(mode, i, t_loss))
    if scheduler is not None:
        scheduler.step()
        print('学习率是{}'.format(opt.state_dict()['param_groups'][0]['lr']))

    return train_loss, val_loss#, pcc


def main():
    parser = argparse.ArgumentParser(description='Semantic decoding')
    parser.add_argument('--model_dir', help='model saved path',
                        default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/semantic/',
                        type=str)
    parser.add_argument('--figure_dir', help='figure saved path',
                        default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/semantic/picture/',
                        type=str)
    parser.add_argument('--batch_size', help='batch size of dnn training', default=64, type=int)
    parser.add_argument('--epoch', help='epoch', default=100, type=int)
    parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
    parser.add_argument('--n_blocks', help='n_mlp_blocks', default=3, type=float)
    parser.add_argument('--k1', help='ratio of loss_CLIP', default=0.01, type=float)
    parser.add_argument('--k2', help='ratio of loss_raw', default=0.5, type=float)
    parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
    parser.add_argument('--warm_up', help='warm_up', default=0, type=float)
    args = parser.parse_args()

    criterion1 = Contrastive_loss()
    criterion2 = nn.MSELoss()

    image_data = torch.tensor(
        np.load('/nfs/nica-datashop/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_images.npy'),
        dtype=torch.float32).to(device)
    f_MRI_data = np.load('/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/activated_voxels_PCA/masked4500_trn_data.npy'.format(args.subj_ID))
    trn_tgt_text = h5py.File('/nfs/nica-datashop/CC2017_for_video_reconstruction/stimuli_frames/Train/Train_captions_new.h5', 'r')[
        'Train_captions']

    model = make_model(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=args.n_blocks, norm_type='ln',
                       act_first=False).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(model_opt, gamma=0.999)

    valset = Val_dataset(f_MRI_data, image_data, trn_tgt_text)

    Train_loss = []
    Val_loss = []
    for epoch in tqdm(range(args.epoch + 1)):

        trainset = Train_dataset(f_MRI_data, image_data, trn_tgt_text)

        model.train()
        train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train_loss, _ = run_epoch(mode='train', data_iter=train_data, model=model,
                                     loss_compute=SimpleLossCompute(criterion1=criterion1, criterion2=criterion2,
                                                                   k1=args.k1, k2=args.k2 ),
                                      norm=1, epoch=epoch,opt=model_opt, scheduler=scheduler)
        Train_loss.append(np.mean(train_loss))

        model.eval()
        val_data = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        _, val_loss= run_epoch(mode='val', data_iter=val_data, model=model,
                                      loss_compute=SimpleLossCompute(criterion1=criterion1, criterion2=criterion2,
                                                                     k1=args.k1, k2=args.k2),
                                      norm=1, epoch=epoch)
        Val_loss.append(np.mean(val_loss))

        if epoch in [20 ,30,40,50, 60,70, 80, 100, 300, 500, 600]: #, 900, 1100, 1300, 1500, 1700, 1900,2500,3000,3500,3999]:
            torch.save(model.state_dict(), os.path.join(args.model_dir.format(args.subj_ID), 'Semantic_decoder_{}.pth'.format(epoch)))

        if epoch in [50,80, 100, 300, 500, 600]:#, 900, 1100, 1300, 1500, 1700, 1900,2500,3000,3500,3999]:
            epochs = range(len(Train_loss))
            plt.plot(epochs, Val_loss, 'g', label='Val loss')
            plt.plot(epochs, Train_loss, 'r', label='Train_loss')
            plt.title('Semantic Decoder')
            plt.legend(loc='upper right')
            plt.savefig(args.figure_dir.format(args.subj_ID) + 'Loss-epoch{}.png'.format(epoch))
            plt.show()



if __name__ == "__main__":
    main()
    print('')

