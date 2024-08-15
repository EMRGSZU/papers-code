import os 
import sys
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from network.CSRN import CSRN
from scipy.io import loadmat
import functools
from setting import Setting
from PIL import Image
from tools.image_utils import rgb2ycbcr, calc_psnr, calc_ssim

def image_padding(img):
    block_size = 32
    hei, wid = img.shape
    if np.mod(hei, block_size) != 0:
        hei_pad = block_size - np.mod(hei, block_size)
    else:
        hei_pad = 0
    if np.mod(wid, block_size) != 0:
        wid_pad = block_size - np.mod(wid, block_size)
    else:
        wid_pad = 0
    pad_img = np.concatenate((img, np.zeros((hei, wid_pad), dtype=np.uint8)), axis=1)
    pad_img = np.concatenate((pad_img, np.zeros((hei_pad, wid + wid_pad), dtype=np.uint8)), axis=0)

    return pad_img, hei, wid, hei_pad, wid_pad

def image_depadding(img, hei_ori, wid_ori):
    img = img[ :hei_ori, :wid_ori]

    return img

def process_img(img, only_y=True):
    n_dim = img.ndim
    if n_dim < 3:
        return img
    else:
        if (img[:,:,0] == img[:,:,1]).all() and (img[:,:,0] == img[:,:,2]).all() and (img[:,:,1] == img[:,:,2]).all():
            return img[:,:,0]
        else:
            img_y = rgb2ycbcr(img, only_y=True)
            return img_y

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = ArgumentParser(description='CSRN')
parser.add_argument('--sample_ratio', type=float, default=0.1, help='sample ratio')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#读取设置文件
config = Setting(args.sample_ratio)

#读取训练集
train_dataset = loadmat(config.train_dataset_name)['train']
transformer = transforms.Compose([transforms.ToTensor()])
trainloader = DataLoader(dataset=train_dataset, batch_size=config.batch, num_workers=0, shuffle=True)

#读取模型
model = CSRN(args.sample_ratio)
model.to(device)  

#优化器和loss，ADAM和L2
optimizer = torch.optim.Adam(model.parameters(), config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.step, gamma = 0.1)
criterion = nn.MSELoss()

best = {'idx_epoch': 0, 'psnr': 0}
trainloss_list = []
psnr_list = []
ssim_list = []

for idx_epoch in range(config.epoch):
    
    #training stage
    model.train()
    n_pic = 0
    train_loss = []
    train_sep_loss = [[] for _ in range(20)]
    for data in trainloader:
        data = data.to(device) 
        rec = model(data)
        n_recurrent = len(rec)
        loss = 0
        loss_list = []
        #就是初始重构和残差重构的loss
        for rec_idx in range(n_recurrent):
            temp = criterion(rec[rec_idx], data)
            loss += temp
            loss_list.append(temp)
        #取个平均
        loss /= n_recurrent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        for i in range(n_recurrent):
            train_sep_loss[i].append(loss_list[i].item())

        n_pic += config.batch
        if n_pic % 1600 == 0:
            output_data = "trainging stage [{}/{} {}] Train Loss: {}".format(idx_epoch, config.epoch, n_pic, loss)
            print(output_data)
            sys.stdout.flush()

    #val
    model.eval()
    psnrs = []
    ssims = []
    times = []
    filelist = os.listdir(config.val_dataset_name)
    with torch.no_grad():
        for i in range(len(filelist)):
            if os.path.splitext(filelist[i])[1] in ['.tif', '.bmp', '.png', '.jpg']:
                name = os.path.splitext(filelist[i])[0]
                filepath = config.val_dataset_name + filelist[i]
                img_ori = Image.open(filepath)
                img_ori = np.array(img_ori)
                img_y = process_img(img_ori, only_y=True)
                img, hei, wid, hei_pad, wid_pad = image_padding(img_y)
                img = img / 255.0
                img = torch.from_numpy(img)
                img = img.type(torch.FloatTensor)
                img = img.view(1, 1, hei + hei_pad, wid + wid_pad)
                img = img.to(device)
                start_time = time.time()
                reconstruction = model(img,)
                end_time = time.time()

                #拿到最终重构图像
                reconstruction = reconstruction[-1][0][0].cpu().data.numpy()
                reconstruction = np.clip(reconstruction, 0, 1)
                reconstruction *= 255
                reconstruction = image_depadding(reconstruction, hei, wid)
            
                psnr = calc_psnr(np.array(np.round(reconstruction), dtype = 'uint8'), img_y)
                ssim = calc_ssim(np.array(np.round(reconstruction), dtype = 'uint8'), img_y)
                cal_time = end_time -start_time
                print(psnr,'  ', ssim, '---{}'.format(cal_time))

                psnrs.append(psnr)
                ssims.append(ssim)
                times.append(cal_time)

    scheduler.step()
    print('mean_psnr = {}, mean_ssim = {}, mean_time = {}'.format(np.mean(psnrs), np.mean(ssims), np.mean(times)))

    mean_train_loss = np.mean(train_loss)
    trainloss_list.append(mean_train_loss)
    mean_psnr = np.mean(psnrs)
    psnr_list.append(mean_psnr)
    mean_ssim = np.mean(ssims)
    ssim_list.append(mean_ssim)

    output_file = open(config.log_file, 'a')
    output_file.write('[{}/{}] \nTrain Loss: {} \nSet11: psnr: {}ssim: {}'.format(idx_epoch + 1, config.epoch, np.mean(mean_train_loss), np.mean(psnrs), np.mean(ssims)) + '\n')
    for i in range(n_recurrent):
        output_file.write('Train: ')
        output_file.write('[{}/{}] {} Loss: {}  '.format(idx_epoch, config.epoch,i, np.mean(train_sep_loss[i])))  
    output_file.write('\n')  
    output_file.close()

    if np.mean(mean_psnr) > best['psnr']:
        best['psnr'] = np.mean(mean_psnr)
        best['idx_epoch'] = idx_epoch
        torch.save(model.state_dict(), config.model_dir + "/model_{}.pkl".format(idx_epoch + 1))  # save only the parameters

output_file = open(config.log_file, 'a')
output_file.write('we aquire best val psnr at epoch {} -- {}.\n'.format(best['idx_epoch'], best['psnr']))

from plot import plot_psnr_ssim
plot_psnr_ssim(psnr_list, ssim_list, trainloss_list, config.epoch, config.analysis)

