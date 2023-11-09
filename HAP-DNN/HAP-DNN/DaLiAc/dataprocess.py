import numpy as np
from numpy import *
import glob
import scipy.stats as st
import statsmodels.tsa.api as smt
from scipy import signal





def split_data1(file_list):
    labels = []
    train_data = []
    for i in range(len(file_list)):
        files = np.loadtxt(file_list[i], dtype=float, delimiter=',', unpack=True) #按列读取数据

        '''将输入数据分段'''
        data_label = files[24]
        data_label = list(data_label)
        g = [data_label[:1]]
        [g[-1].append(y) if x == y else g.append([y]) for x, y in zip(data_label[:-1], data_label[1:])]
        len_cnt = 0
        data_list = []
        for data_len in range(len(g)):
            data_list.append(files[:,len_cnt:len_cnt + len(g[data_len])])
            len_cnt += len(g[data_len])

        '''滑动窗口提取特征，窗口大小为5s，采样频率205hz，即1025个点，窗口重叠率50%'''
        for data_idx in range(len(data_list)):
            datas = data_list[data_idx]
            for j in range(0, len(datas[0]), 512):
                data = datas[:, j:j+1025]
                if(len(data[0])<1025): #数据长度小于1025直接丢掉不用
                    continue
                data_channels = []
                label = [0] * 13
                label[int(g[data_idx][0])-1] = 1
                for chans_idx in range(0, len(data)-1):
                    data_temp = []
                    '''特征提取'''

                    data_temp.append(np.min(data[chans_idx]))
                    data_temp.append(np.max(data[chans_idx]))
                    data_temp.append(np.mean(data[chans_idx]))
                    data_temp.append(np.var(data[chans_idx]))
                    data_temp.append(st.skew(data[chans_idx]))
                    data_temp.append(st.kurtosis(data[chans_idx]))

                    fftlist = np.abs(np.fft.fft(data[chans_idx], 25))
                    fft_rank = np.argsort(fftlist)[:-1]
                    fft_peak = fftlist[fft_rank[:5]]
                    for idx in fft_peak:
                        data_temp.append(idx)
                    '''归一化'''
                    min_data = np.min(data_temp)
                    max_data = np.max(data_temp)
                    for x in range(len(data_temp)):
                        data_temp[x] = (data_temp[x] - min_data) / (max_data - min_data)

                    data_channels.append(np.array(data_temp, dtype=float))
                train_data.append(np.array(data_channels, dtype = float))
                labels.append(np.array(label, dtype = float))
                print("one sample finish:",j//1025)
        print("one man finish:",i)

    '''shuffle'''
    train_data = np.array(train_data, dtype = float, ndmin = 3)
    labels = np.array(labels, dtype=float, ndmin=2)
    shuffle_idx = np.random.permutation(train_data.shape[0])
    shuffle_data = train_data[shuffle_idx, :, :]
    shuffle_labels = labels[shuffle_idx, :]
    print(train_data.shape)
    print(labels.shape)
    return shuffle_data, shuffle_labels

path = './data/*'
f_list = glob.glob(path)
train_data, train_label = split_data1(f_list)
np.savez("daliac2.npz", data = train_data, label = train_label)



