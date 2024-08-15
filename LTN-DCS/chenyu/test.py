import os
import torch
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from network.CSRN import CSRN
from tools.image_utils import rgb2ycbcr, calc_psnr, calc_ssim
from setting import Setting
import hiddenlayer as h
print(torch.__version__)

# def image_padding(img):
#     block_size = 32
#     hei, wid = img.shape
#     if np.mod(hei, block_size) != 0:
#         hei_pad = block_size - np.mod(hei, block_size)
#     else:
#         hei_pad = 0
#     if np.mod(wid, block_size) != 0:
#         wid_pad = block_size - np.mod(wid, block_size)
#     else:
#         wid_pad = 0
#     pad_img = np.concatenate((img, np.zeros((hei, wid_pad), dtype=np.uint8)), axis=1)
#     pad_img = np.concatenate((pad_img, np.zeros((hei_pad, wid + wid_pad), dtype=np.uint8)), axis=0)

#     return pad_img, hei, wid, hei_pad, wid_pad

# def image_depadding(img, hei_ori, wid_ori):
#     img = img[:, :, :hei_ori, :wid_ori]

#     return img

def image_padding(img):
    block_size = 32
    hei, wid = img.shape
    hei_blk = hei // 32
    wid_blk = wid // 32

    pad_img = img[:hei_blk * 32, :wid_blk * 32]

    return pad_img
    # block_size = 32
    # hei, wid = img.shape
    # if np.mod(hei, block_size) != 0:
    #     hei_pad = block_size - np.mod(hei, block_size)
    # else:
    #     hei_pad = 0
    # if np.mod(wid, block_size) != 0:
    #     wid_pad = block_size - np.mod(wid, block_size)
    # else:
    #     wid_pad = 0
    # pad_img = np.concatenate((img, np.zeros((hei, wid_pad), dtype=np.uint8)), axis=1)
    # pad_img = np.concatenate((pad_img, np.zeros((hei_pad, wid + wid_pad), dtype=np.uint8)), axis=0)
    #
    # return pad_img, hei, wid, hei_pad, wid_pad

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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sample_ratio = 0.1
config = Setting(sample_ratio)

modelpath = config.model_dir
logpath = config.log_file
datasetpath = '/data/majunnan/machi/test_set/'
picpath = config.pic_dir
if os.path.exists(picpath) == False:
    os.makedirs(picpath)
device = 'cuda'

model = CSRN(sample_ratio)

# vis_graph = h.build_graph(model, torch.zeros([1 ,1, 256, 256]))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("demo1.png")   # 保存图像的路径

modelname = modelpath + '/model_81.pkl'
model.load_state_dict(torch.load(modelname))
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')


# datasets = ['Set11', 'Set5', 'DIV2K_valid_HR', 'Set14', 'BSDS100', 'Urban100', ]
datasets = ['Set14' ]
# datasets = ['DIV2K_valid_HR']

psnr_list = []
ssim_list = []
time_list = []

model.eval()
for dataset in datasets:
    save_dir = config.pic_dir + '/{}/'.format(dataset)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    dataset_path = datasetpath + dataset + '/'
    if os.path.exists(dataset_path):
        rootpath = picpath + '/'
        if os.path.exists(rootpath) == False:
            os.makedirs(rootpath)
        rootpath = rootpath + dataset + '/'
        if os.path.exists(rootpath) == False:
            os.makedirs(rootpath)
        
        psnrs = []
        ssims = []
        times = []

        filelist = os.listdir(dataset_path)
        with torch.no_grad():
            for i in range(len(filelist)):
                i = i+1
                if os.path.splitext(filelist[i])[1] in ['.tif', '.bmp', '.png', '.jpg']:
                    name = os.path.splitext(filelist[i])[0]
                    filepath = dataset_path + filelist[i]
                    img_ori = Image.open(filepath)
                    img_ori = np.array(img_ori)
                    img_y = process_img(img_ori, only_y=True)

                    # img_y = Image.fromarray(np.array(img_y, dtype='uint8'))
                    # img_y.save('Gray.png')

                    img_y  = image_padding(img_y)
                    img = img_y / 255.0
                    img = torch.from_numpy(img)
                    img = img.type(torch.FloatTensor)
                    img = img.unsqueeze(0).unsqueeze(0)
                    img = img.to(device)
                    start_time = time.time()
                    predictions = model(img,)
                    end_time = time.time()
                    time_consume = end_time - start_time
                    print(time_consume)

                    n_rec = len(predictions)
                    
                    predictions_list = []
                    for prediction in predictions:
                        prediction = prediction.cpu().data.numpy()
                        prediction = np.clip(prediction, 0, 1)
                        prediction *= 255
                        predictions_list.append(prediction)
                    
                    temp_psnrs = []
                    temp_ssims = []
                    for prediction in predictions_list:
                        temp_psnr = calc_psnr(np.array(np.round(prediction[0][0]), dtype = 'uint8'), img_y)
                        temp_ssim = calc_ssim(np.array(np.round(prediction[0][0]), dtype = 'uint8'), img_y)
                        print(temp_psnr,'  ', temp_ssim)
                        temp_psnrs.append(temp_psnr)
                        temp_ssims.append(temp_ssim)
                    psnrs.append(temp_psnrs)
                    ssims.append(temp_ssims)
                    times.append(time_consume)

                    cur_psnr = temp_psnrs[-1]
                    cur_ssim = temp_ssims[-1]

                    time_list.append(time_consume)
                    psnr_list.append(cur_psnr)
                    ssim_list.append(cur_ssim)

                    save_path = config.pic_dir + '/{}/'.format(dataset) + name + '_{}_{}'.format(cur_psnr, cur_ssim) + '.png'
                    img = Image.fromarray(np.array(np.round(predictions_list[-1][0][0]), dtype = 'uint8')) 
                    img.save(save_path)

            psnr_dict = dict()
            ssim_dict = dict()
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            for i in range(n_rec):
                i_psnr = psnrs[:, i]
                i_psnr = np.mean(i_psnr)
                psnr_dict['{}'.format(i)] = round(i_psnr, 2)
                i_ssim = ssims[:, i]
                i_ssim = np.mean(i_ssim)
                ssim_dict['{}'.format(i)] = round(i_ssim, 4)

            with open(logpath, 'a+') as f:
                f.write(dataset + '\n')
                f.write(str(psnr_dict) + '\n')
                f.write(str(ssim_dict) + '\n')
                f.write(str(np.mean(times)) + '\n')

print(np.mean(time_list))
print(np.mean(psnr_list))
print(np.mean(ssim_list))
# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

with open(logpath, 'a+') as f:
    f.write(str(np.mean(time_list)) + '\n')
    f.write(str(np.mean(psnr_list)) + '\n')
    f.write(str(np.mean(ssim_list)) + '\n')
    f.write(str(total_params) + 'total parameters.' + '\n')