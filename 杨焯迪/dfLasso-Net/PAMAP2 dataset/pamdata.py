import numpy as np
import os
import glob
import scipy.stats as st
import statsmodels.tsa.api as smt
'''
PAMAP2数据集：18个类别如下：
– 1 lying – 2 sitting – 3 standing – 4 walking – 5 running – 6 cycling – 7 Nordic walking – 9 watching TV
– 10 computer work – 11 car driving – 12 ascending stairs – 13 descending stairs – 16 vacuum cleaning
– 17 ironing – 18 folding laundry – 19 house cleaning – 20 playing soccer – 24 rope jumping
– 0 other (transient activities)//不需要，直接去掉
如果想要18类活动，必须要用上optional的数据才行

每个文件有54列，其中：– 1 timestamp (s) – 2 activityID 
– 3 heart rate (bpm)– 4-20 IMU hand – 21-37 IMU chest – 38-54 IMU ankle
第2列做label，并使用后面的传感器数据
由于每个单元有17列，在这17列中，只需要用到2-4 3D-acceleration data,8-10 3D-gyroscope data,11-13 3D-magnetometer data即可
综上所述，需要用到的列为：（从0开始编号）
1:label 4-6,10-12,13-15, 21-23,27-29,30-32, 38-40,44-46,47-49, 分别对应hand,chest,ankle的acc,gyro,magn计
关于NaN处理：直接复制前一个值即可
采样频率为100hz，样本分割以5s为滑动窗口，窗口重叠率50%
'''


def split_data1(file_list):#不进行特征提取
    labels = []
    train_data = []
    for i in range(len(file_list)):
        files = np.loadtxt(file_list[i], dtype=float, delimiter=' ', unpack=True)  # 按列读取数据
        data_idx = [p for p in range(4, 7)] + [p for p in range(10, 16)]
        data_idx = data_idx + [p for p in range(21, 24)] + [p for p in range(27, 33)]
        data_idx = data_idx + [p for p in range(38, 41)] + [p for p in range(44, 50)]
        data_input = files[data_idx]
        data_label = files[1]
        for j in range(0, len(data_input[0]), 250):
            data_channels = []
            if j >= len(data_input[0] - 500):
                break
            else:
                data = data_input[:, j:j + 500]  # 每个样本要处理的数据
                label_temp = data_label[j:j + 500]
                '''先统计label出现次数，次数最多的为样样本label；如果label值为0，就直接丢弃，因为不需要该类别'''
                lab_idx = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24, 0]
                lab_cnt = []
                for l in lab_idx:
                    lab_cnt.append(np.sum(label_temp == l))
                lab_rank = np.argsort(lab_cnt)  # 降序排序，找到最多的类别
                label = [0] * 18
                if (lab_rank[18] == 18):  # 如果标志为0的类别最多，就丢弃该样本
                    continue
                else:
                    label[lab_rank[18]] = 1  # 对应位置label为1
                    for chans in data:
                        # 先将值为nan的地方找出来并处理
                        for k in range(len(chans)):
                            if np.isnan(chans[k]):
                                if k!=0:
                                    chans[k]=chans[k-1]
                                else:
                                    chans[k]=0
                        data_channels.append(chans)
                    data_channels = np.array(data_channels, dtype=float)
                    if data_channels.shape[0]==27 and data_channels.shape[1]==500:
                        train_data.append(data_channels)
                        labels.append(np.array(label, dtype=float))
                print("one sample finish", j // 500)
        print("one man finish", i)
    train_data = np.array(train_data, dtype=float, ndmin=3)
    labels = np.array(labels, dtype=float, ndmin=2)
    print(train_data.shape)
    print(labels.shape)
    return train_data, labels

def split_data(file_list):
    labels = []
    train_data = []
    for i in range(len(file_list)):
        files = np.loadtxt(file_list[i], dtype = float, delimiter = ' ', unpack = True)#按列读取数据
        data_idx = [p for p in range(4,7)]+[p for p in range(10,16)]
        data_idx = data_idx + [p for p in range(21,24)]+[p for p in range(27,33)]
        data_idx = data_idx + [p for p in range(38,41)]+[p for p in range(44,50)]
        data_input = files[data_idx]
        data_label = files[1]
        for j in range(0,len(data_input[0]),250):
            data_channels = []
            if j>=len(data_input[0]-500):
                break
            else:
                data = data_input[:,j:j+500]#每个样本要处理的数据
                label_temp = data_label[j:j+500]
                '''先统计label出现次数，次数最多的为样样本label；如果label值为0，就直接丢弃，因为不需要该类别'''
                lab_idx = [1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,24,0]
                lab_cnt = []
                for l in lab_idx:
                    lab_cnt.append(np.sum(label_temp==l))
                lab_rank = np.argsort(lab_cnt)#降序排序，找到最多的类别
                label = [0]*18
                if(lab_rank[18]==18):#如果标志为0的类别最多，就丢弃该样本
                    continue
                else:
                    label[lab_rank[18]]=1#对应位置label为1
                    for chans in data:
                        # 先将值为nan的地方找出来并处理
                        data_temp = []
                        for k in range(len(chans)):
                            if np.isnan(chans[k]):
                                if k!=0:
                                    chans[k]=chans[k-1]
                                else:
                                    chans[k]=0
                        '''特征提取'''
                        data_temp.append(np.min(chans))
                        data_temp.append(np.max(chans))
                        data_temp.append(np.mean(chans))
                        data_temp.append(np.var(chans))
                        data_temp.append(st.skew(chans))
                        data_temp.append(st.kurtosis(chans))
                        '''计算傅里叶变换的五个峰值'''
                        fftlist = np.abs(np.fft.fft(chans, 25))
                        fft_rank = np.argsort(fftlist)[:-1]
                        fft_peak = fftlist[fft_rank[:5]]
                        for idx in fft_peak:
                            data_temp.append(idx)
                        '''归一化'''
                        min_data = np.min(data_temp)
                        max_data = np.max(data_temp)
                        for x in range(len(data_temp)):
                            data_temp[x] = (data_temp[x] - min_data)/(max_data - min_data)

                        data_channels.append(np.array(data_temp, dtype = float))
                    labels.append(np.array(label,dtype=float))
                    train_data.append(np.array(data_channels,dtype=float))
                print("one sample finish",j//500)
        print("one man finish",i)
    train_data = np.array(train_data,dtype=float,ndmin=3)
    labels = np.array(labels,dtype=float,ndmin=2)
    print(train_data.shape)
    print(labels.shape)
    return train_data, labels



path='./data/PAM/*'
f_list=glob.glob(path)
train_data,train_label = split_data1(f_list)
np.savez("pamapo.npz",data=train_data,label=train_label)
