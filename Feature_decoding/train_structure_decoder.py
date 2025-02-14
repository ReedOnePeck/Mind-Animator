import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14
from MindAnimator.models.data_aug import fMRI_sparsification
from MindAnimator.models.Decoder import Strcuture_decoder
from MindAnimator.models.utils import decay
from torch.optim import lr_scheduler
from einops import rearrange
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda:0')



def make_model(in_dim=4500, out_dim=4*64*64, h=512, n_blocks=3, norm_type='ln', act_first=False):
    model = Strcuture_decoder(in_dim=in_dim, out_dim=out_dim, h=h, n_blocks=n_blocks, norm_type=norm_type,
                         act_first=act_first)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Train_dataset(Dataset):
    def __init__(self, fMRI_trn, first_frames_train):
        self.fMRI = torch.tensor(fMRI_sparsification(data=fMRI_trn,rate1=0.4 , rate2=0.5), dtype=torch.float32)[:4000,:].to(device)
        self.frames = first_frames_train[:4000,:]

    def __len__(self):
        data_size = 4000
        return data_size

    def __getitem__(self, item):
        trn_fMRI = self.fMRI[item:item + 1, :]
        trn_frame = self.frames[item:item + 1, :]

        return trn_fMRI, trn_frame



class Val_dataset(Dataset):
    def __init__(self, fMRI_trn, first_frames_train):
        self.fMRI = torch.tensor(fMRI_sparsification(data=fMRI_trn, rate1=0.4, rate2=0.5), dtype=torch.float32)[4000:,:].to(device)
        self.frames = first_frames_train[4000:, :]

    def __len__(self):
        data_size = 320
        return data_size

    def __getitem__(self, item):
        trn_fMRI = self.fMRI[item:item + 1, :]
        trn_frame = self.frames[item:item + 1, :]

        return trn_fMRI, trn_frame

# ----------------------------------------------------------Train------------------------------------------------------------------
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.01

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))



class SimpleLossCompute:
    def __init__(self, criterion, k,  opt=None):
        self.criterion = criterion
        self.k = k
        self.opt = opt

    def __call__(self, mode, epoch=None,y_pred=None, y_tgt=None,regularization=None, weight_decay = None):
        if mode == 'train':
            loss_proj = self.criterion(y_pred, y_tgt)
            loss =  self.k * loss_proj + weight_decay * regularization

            return loss, loss_proj.data.item()

        if mode == 'val':
            loss_proj = self.criterion(y_pred, y_tgt)
            loss = self.k * loss_proj

            return  loss.data.item()


def run_epoch(mode, data_iter, model, loss_compute, epoch=None, opt=None, scheduler=None, weight_decay = None):
    val_loss = []
    train_loss = []

    for i, batch in tqdm(enumerate(data_iter)):
        if mode == 'train':
            loss_reg = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    loss_reg += torch.sum(torch.abs(param))

            trn_fMRI, trn_frame = batch
            y_pred = model.forward(x = trn_fMRI.squeeze(1))
            loss, loss_proj = loss_compute(mode=mode, epoch=epoch, y_pred=y_pred, y_tgt=trn_frame, regularization=loss_reg, weight_decay = weight_decay)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss_proj)
            if i % 10 == 0:
                print('当前状态为{},进行到第{}个batch，这一个batch的损失函数为{}'.format(mode, i, loss_proj ))

        if mode == 'val':
            val_fMRI, val_frame = batch
            y_pred = model.forward(x=val_fMRI.squeeze(1))
            t_loss = loss_compute(mode=mode, epoch=epoch, y_pred=y_pred, y_tgt=val_frame)

            val_loss.append(t_loss)
            if i % 10 == 0:
                print('当前状态为{},进行到第{}个batch，这一个batch的损失函数为{}'.format(mode, i, t_loss))
    if scheduler is not None:
        scheduler.step()
        print('学习率是{}'.format(opt.state_dict()['param_groups'][0]['lr']))

    return train_loss, val_loss

def main():
    parser = argparse.ArgumentParser(description='strcuture decoding')
    parser.add_argument('--model_dir', help='model saved path', default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/strcuture/',type=str)
    parser.add_argument('--figure_dir', help='figure saved path', default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/strcuture/picture/',type=str)
    parser.add_argument('--batch_size', help='batch size of dnn training', default=64, type=int)
    parser.add_argument('--epoch', help='epoch', default=400, type=int)
    parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
    parser.add_argument('--n_blocks', help='n_mlp_blocks', default=2, type=float)
    parser.add_argument('--k', help='ratio of loss_CLIP', default=1, type=float)
    parser.add_argument('--lr', help='learning rate', default=0.000001, type=float)
    parser.add_argument('--warm_up', help='warm_up', default=50, type=float)
    args = parser.parse_args()


    fMRI_trn = np.load(
        '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/activated_voxels_PCA/masked4500_trn_data.npy'.format(
            args.subj_ID))

    video_path_train = '/nfs/nica-datashop/CC2017_for_video_reconstruction/stimuli_clips/Train/contents_float32.npy'
    frames_train = rearrange(torch.tensor(np.load(video_path_train), dtype=torch.float32),
                             "b c f h w -> b f (c h w)").to(device)
    first_frames_train = frames_train[:, 0, :]

    criterion = nn.MSELoss()
    model = make_model(in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=args.n_blocks, norm_type='ln', act_first=False).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(model_opt, gamma=0.999)

    valset = Val_dataset(fMRI_trn, first_frames_train)
    Train_loss = []
    Val_loss = []
    weight_decay = 4e-6
    decay_rate = 0.00005


    for epoch in tqdm(range(args.epoch +1 )):
        model.train()
        trainset = Train_dataset(fMRI_trn, first_frames_train)
        train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train_loss, _ = run_epoch(mode='train', data_iter=train_data, model=model,
                                  loss_compute=SimpleLossCompute(criterion=criterion, k=args.k),
                                  epoch=epoch, opt=model_opt, scheduler=scheduler, weight_decay = weight_decay)
        Train_loss.append(np.mean(train_loss))
        weight_decay = decay(weight_decay, decay_rate, epoch)


        model.eval()
        val_data = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
        _, val_loss = run_epoch(mode='val', data_iter=val_data, model=model,
                                 loss_compute=SimpleLossCompute(criterion=criterion, k=args.k),epoch=epoch)
        Val_loss.append(np.mean(val_loss))


        if epoch in [20 ,30,40,50, 60,70, 80, 100, 300, 500, 600]: #, 900, 1100, 1300, 1500, 1700, 1900,2500,3000,3500,3999]:
            torch.save(model.state_dict(), os.path.join(args.model_dir.format(args.subj_ID), 'Structure_decoder_{}.pth'.format(epoch)))

        if epoch in [100, 200, 300, 400]:
            epochs = range(len(Train_loss))
            plt.plot(epochs, Val_loss, 'g', label='Val loss')
            plt.plot(epochs, Train_loss, 'r', label='Train_loss')
            plt.title('Structure Decoder')
            plt.legend(loc='upper right')
            plt.savefig(args.figure_dir + 'Loss-epoch{}.png'.format(epoch))
            plt.show()


if __name__ == "__main__":
    main()
    print('')

