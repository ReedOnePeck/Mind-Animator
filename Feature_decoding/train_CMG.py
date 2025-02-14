import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from MindAnimator.models.Decoder import make_CMG_model, subsequent_mask
from MindAnimator.models.utils import decay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14
device = torch.device('cuda:2')




class Train_dataset(Dataset):
    def __init__(self,trn_src_path, trn_tgt_path,  mask_ratio, pad = 0):
        self.trn_src = torch.tensor(np.load(trn_src_path)[:4000,:], dtype=torch.float32).to(device)

        trn_tgt = torch.tensor(np.load(trn_tgt_path), dtype=torch.float32)
        swapped_data = rearrange(trn_tgt, "b c f h w -> b f (c h w)")
        self.trg_input = swapped_data[:4000, :-1 ,: ].to(device)
        self.trg_output = swapped_data[:4000, 1:, :].to(device)
        trg_ = torch.from_numpy(np.ones(((self.trn_src).size(0), swapped_data.size(1)))[:4000, :-1])
        self.trg_mask = self.make_std_mask(trg_, pad, mask_ratio).to(device)

    @staticmethod
    def make_std_mask(tgt, pad, mask_ratio):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1), mask_ratio).type_as(tgt_mask.data))
        return tgt_mask

    def __len__(self):
        data_size = 4000
        return data_size

    def __getitem__(self, item):
        trn_src = self.trn_src[item, :]
        trg_input = self.trg_input[item, :, :]
        trg_output = self.trg_output[item, :, :]
        trg_mask = self.trg_mask[item, :, :]

        return trn_src, trg_input, trg_output, trg_mask




class Val_dataset(Dataset):
    def __init__(self, trn_src_path, trn_tgt_path,):
        self.trn_src = torch.tensor(np.load(trn_src_path)[4000:, :], dtype=torch.float32).to(device)
        trn_tgt = torch.tensor(np.load(trn_tgt_path), dtype=torch.float32)
        swapped_data = rearrange(trn_tgt, "b c f h w -> b f (c h w)")
        self.trg_output = swapped_data[4000:, 1:, :].to(device)
        self.trg_input = swapped_data[4000:, :2, :].to(device)

    def __len__(self):
        data_size = 320
        return data_size

    def __getitem__(self, item):
        trn_src = self.trn_src[item, :]
        trg_input = self.trg_input[item, :, :]
        trg_output = self.trg_output[item, :, :]
        return trn_src, trg_output, trg_input



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
    def __init__(self, criterion1, criterion2, k1, k2, opt=None):
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.k1 = k1
        self.k2 = k2
        self.opt = opt

    def __call__(self,mode, x, y, frame1_2, loss_reg=None, weight_decay=None, norm=1):
        if mode == 'train':
            loss_content = self.criterion1(x, y)
            loss_flow = 0
            loss = self.k1 * loss_content + self.k2 * loss_flow + weight_decay* loss_reg
            print("loss_content",self.k1 * loss_content)
            print("loss_reg", weight_decay* loss_reg)

            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return self.k1 * loss_content.data.item(), loss_flow, loss.data.item()

        if mode == 'val':
            loss_content = self.criterion1(x, y)
            loss_flow = 0
            loss = self.k1 * loss_content + self.k2 * loss_flow #+ weight_decay * loss_reg

            return self.k1 * loss_content.data.item(), loss_flow, loss.data.item()




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

def run_epoch(mode, data_iter, model, loss_compute, weight_decay=None,mask_ratio = None, epoch = None, subj_ID=None):
    train_loss = []; train_content_loss = []; train_flow_loss = []
    val_loss = []; val_content_loss = []; val_flow_loss = []

    for i, batch in tqdm(enumerate(data_iter)):
        if mode == 'train':
            loss_reg = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    loss_reg += torch.sum(torch.abs(param))

            src_mask = None
            trn_src, trg_input, trg_output, trg_mask = batch
            out = model.forward(trn_src, trg_input, src_mask, trg_mask)

            loss_content, loss_flow, loss = loss_compute(mode=mode, x=out, y=trg_output, frame1_2 = trg_input[:,:2,:], loss_reg =loss_reg, weight_decay =weight_decay)
            train_loss.append(loss)
            train_content_loss.append(loss_content)
            train_flow_loss.append(loss_flow)
            if i % 20 == 1:
                print("Training Epoch Step: %d MSE Loss per batch: %f " %(i, loss))
        if mode == 'val':
            src_mask = None
            val_src, val, val_frame_1_2 = batch
            out = greedy_decode(first_frame=val_frame_1_2[:,0:1,:],model=model,src=val_src,src_mask=src_mask, max_len=8 ,mask_ratio=0 )[:,1:,:]#).cpu().detach().numpy()
            loss_content, loss_flow, loss = loss_compute(mode=mode, x=out, y=val, frame1_2=val_frame_1_2)
            val_loss.append(loss)
            val_content_loss.append(loss_content)
            val_flow_loss.append(loss_flow)
    if mode == 'val':
        print('第{}个epoch在测试集上的loss为{},content_loss为{},flow_loss为{}'.format(epoch, np.mean(val_loss), np.mean(val_content_loss), np.mean(val_flow_loss)))

    return train_loss, train_content_loss, train_flow_loss, val_loss, val_content_loss, val_flow_loss



def main():
    parser = argparse.ArgumentParser(description='Transformer based content decoding')
    parser.add_argument('--model_dir', help='model saved path', default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/CMG/', type=str)
    parser.add_argument('--figure_dir', help='figure saved path', default='/data0/home/luyizhuo/NIPS2024实验材料/多个被试实验结果/sub0{}/pretrained_weights/CMG/picture/', type=str)
    parser.add_argument('--batch_size', help='batch size of dnn training', default=64, type=int)
    parser.add_argument('--epoch', help='epoch', default=300, type=int)
    parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
    parser.add_argument('--mask_ratio', help='mask_ratio of Sparse Causal mask ', default=0.6, type=float)
    parser.add_argument('--k1', help='ratio of loss_PCA', default=1, type=float)
    parser.add_argument('--k2', help='ratio of loss_raw', default=1, type=float)
    parser.add_argument('--lr', help='learning rate', default=0.00004, type=float)
    parser.add_argument('--N', help='number of modules', default=2, type=int)
    parser.add_argument('--warm_up', help='warm_up', default=50, type=float)
    parser.add_argument('--d_model', help='d_model', default=4096*4, type=float)
    args = parser.parse_args()



    trn_src_path = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/fMRI_in_fslr/sub0{}/Train/activated_voxels_PCA/masked4500_trn_data.npy'.format(args.subj_ID)
    trn_tgt_path = '/nfs/nica-datashop/CC2017_for_video_reconstruction/stimuli_clips/Train/contents_float32.npy'

    if not os.path.exists(args.figure_dir):
        os.makedirs(args.model_dir)
        os.makedirs(args.figure_dir)
    trainset = Train_dataset(trn_src_path=trn_src_path, trn_tgt_path =trn_tgt_path , mask_ratio = args.mask_ratio, pad = 0)
    valset = Val_dataset(trn_src_path=trn_src_path, trn_tgt_path =trn_tgt_path )

    val_data = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    model = make_CMG_model(N=args.N, d_model=args.d_model, d_ff=768, h=4, dropout=0.2).to(device)
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, args.warm_up,
                        torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9))

    weight_decay = 0.0000006
    decay_rate = 0.001

    Train_loss = []
    Train_content_loss = []
    Val_loss = []
    Val_content_loss = []

    for epoch in tqdm(range(args.epoch+1)):
        train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        model.train()

        train_loss, train_content_loss, _, _, _, _ = run_epoch(mode='train', data_iter=train_data, model=model,
                  loss_compute=SimpleLossCompute(criterion1=criterion1, criterion2=criterion2, k1=args.k1, k2=args.k2,opt=model_opt),
                  weight_decay=weight_decay, mask_ratio = args.mask_ratio, epoch = epoch)

        weight_decay = decay(weight_decay, decay_rate, epoch)

        Train_loss.append(np.mean(train_loss))
        Train_content_loss.append(np.mean(train_content_loss))

        model.eval()
        _, _, _, val_loss, val_content_loss, _ = run_epoch(mode='val', data_iter=val_data, model=model,
                  loss_compute=SimpleLossCompute(criterion1=criterion1, criterion2=criterion2, k1=args.k1, k2=args.k2,opt=model_opt),
                  weight_decay=weight_decay, mask_ratio = 0, epoch = epoch)

        Val_loss.append(np.mean(val_loss))
        Val_content_loss.append(np.mean(val_content_loss))

        if epoch in [50,100,150,200,300,500]:
           torch.save(model.state_dict(), os.path.join(args.model_dir.format(args.subj_ID), 'model_{}.pth'.format(epoch)))

        if epoch in [50, 70, 100, 150, 200, 300, 400, 500, 600, 800, 999]:
            epochs = range(len(Train_loss))

            plt.figure()
            plt.plot(epochs, Train_content_loss, 'r', label='Train_loss')
            plt.plot(epochs, Val_content_loss, 'g', label='Val_loss')
            plt.title('Consistency Motion Generator')
            plt.legend(loc='upper right')
            plt.savefig(args.figure_dir.format(args.subj_ID) + 'CMG_loss-Epoch{}.png'.format(epoch))
            plt.show()




if __name__ == "__main__":
    main()
    print('')


