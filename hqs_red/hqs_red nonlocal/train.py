import os 
import sys
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from network.network import ADMM_RED_UNFOLD
from network.network_pre import Pre
from scipy.io import loadmat
from setting import Setting
from tools.image_utils import rgb2ycbcr, calc_psnr, calc_ssim
from PIL import Image

def image_padding(img):
    """
    保证输入图像大小是32的倍数，如输入为500*500，则补零至512*512
    """
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
    """
    把补零的边去掉
    """
    img = img[:, :, :hei_ori, :wid_ori]

    return img

def process_img(img, only_y=True):
    """
    主要功能是为图片补零并提取图像的Y通道
    """
    n_dim = img.ndim
    if n_dim < 3:
        return img
    else:
        if (img[:,:,0] == img[:,:,1]).all() and (img[:,:,0] == img[:,:,2]).all() and (img[:,:,1] == img[:,:,2]).all():
            return img[:,:,0]
        else:
            img_y = rgb2ycbcr(img, only_y=True)
            return img_y

"""
选择GPU
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

"""
一些设置
"""
parser = ArgumentParser(description='CSRN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--sample_ratio', type=float, default=0.3, help='sample ratio')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
sample_ratio = args.sample_ratio

config = Setting(args.sample_ratio)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

"""
准备数据
"""
train_dataset = loadmat(config.train_dataset_name)['train']
transformer = transforms.Compose([transforms.ToTensor()])
trainloader = DataLoader(dataset=train_dataset, batch_size=config.batch, num_workers=0, shuffle=True)
 
"""
加载模型
"""
model = ADMM_RED_UNFOLD(args.sample_ratio)
# model_pre = Pre(args.sample_ratio)
# model_pre.load_state_dict(torch.load('/data/machi/algorithms/compressed_sensing/my_work/deepunfold_red/deepUnfold_ADMM_RED20/0.3/model/model_pre_45.pkl'))
# model.init.weight = model_pre.init.weight
# model.sample.weight = model_pre.sample.weight

model.to(device)

if start_epoch >= 1:
    model.load_state_dict(torch.load('./results/net_params_{}_{}.pkl'.format(sample_ratio, start_epoch)))    


"""
设置优化器
"""
optimizer = torch.optim.Adam(model.parameters(), config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.step, gamma = 0.1)
criterion = nn.MSELoss() #L2 Loss, 不过L1 Loss其实会更好些

best = {'idx_epoch': 0, 'psnr': 0}
trainloss_list = []
psnr_list = []
ssim_list = []

last = None
for idx_epoch in range(config.epoch):
    
    #训练
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
        for rec_idx in range(n_recurrent):
            temp = criterion(rec[rec_idx], data)
            loss += temp
            loss_list.append(temp)
        loss /= n_recurrent

        last = rec[rec_idx].clone()

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

    #验证
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
                img = img_y / 255.0
                img = torch.from_numpy(img)
                img = img.type(torch.FloatTensor)
                img = img.unsqueeze(0).unsqueeze(0)
                img = img.to(device)
                start_time = time.time()
                reconstruction = model(img,)
                end_time = time.time()
                
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

    #输出本次的一些记录
    print('mean_psnr = {}, mean_ssim = {}, mean_time = {}'.format(np.mean(psnrs), np.mean(ssims), np.mean(times)))

    mean_train_loss = np.mean(train_loss)
    trainloss_list.append(mean_train_loss)
    mean_psnr = np.mean(psnrs)
    psnr_list.append(mean_psnr)
    mean_ssim = np.mean(ssims)
    ssim_list.append(mean_ssim)

    output_file = open(config.log_file, 'a')
    output_file.write('[{}/{}] \nTrain Loss: {} \nBSDS500: psnr: {}ssim: {}'.format(idx_epoch + 1, config.epoch, np.mean(mean_train_loss), np.mean(psnrs), np.mean(ssims)) + '\n')
    for i in range(n_recurrent):
        output_file.write('Train: ')
        output_file.write('[{}/{}] {} Loss: {}  '.format(idx_epoch, config.epoch,i, np.mean(train_sep_loss[i])))  
    output_file.write('\n')  
    output_file.close()

    #若这一次迭代的效果比之前的更好，就保存模型，不过其实可以直接选择最优的来保存
    if np.mean(mean_psnr) > best['psnr']:
        best['psnr'] = np.mean(mean_psnr)
        best['idx_epoch'] = idx_epoch
        torch.save(model.state_dict(), config.model_dir + "/model_{}.pkl".format(idx_epoch + 1))  # save only the parameters

#记录最好结果
output_file = open(config.log_file, 'a')
output_file.write('we aquire best val psnr at epoch {} -- {}.\n'.format(best['idx_epoch'], best['psnr']))

# from plot import plot_psnr_ssim
# plot_psnr_ssim(psnr_list, ssim_list, trainloss_list, config.epoch, config.analysis)