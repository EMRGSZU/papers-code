import numpy as np
import os
import glob
import scipy.stats as st
import statsmodels.tsa.api as smt

'''
OPPORTUNITY dataset
对该数据集进行处理，使用到的通道编号如下：
    2-37：对应12个3D加速度传感器
    38-117，119-133：对应7个惯性单元
    232-243：LOC TAG 1-4
一共143个通道。
分类标签：使用244 locomotion，分为四类：stand,walk,sit,lie
按照500ms大小窗口，步长为250ms，后面根据244来判断是否保留该样本
提取特征如下：
（1）最小值，最大值，均值，偏度和峰度。
（2）对应傅里叶变换的五个峰值。
（3）11个自相关函数样本。
一个样本对应21*143=3003个特征。
对于样本中途有NAN值的，使用两种方法处理数据：
（1）针对缺失一半以上数据的传感器，直接剔除相应处理器进行进一步处理。
（2）缺失值少于一半时，简单的重复之前的值来代替缺失值。（如何重复？）
'''

def split_data(file_list):
    labels = []
    train_data = []
    for i in range(len(file_list)-1):
        files = np.loadtxt(file_list[i], dtype = float, delimiter = ' ', unpack = True) #按列读取数据
        '''使用通道编号和数组下标差1，编号为：2-117，119-133，232-243，标签'''
        data_idx = [p for p in range(1,117)] + [p for p in range(118,133)] + [p for p in range(231,243)]
        data_input = files[data_idx]#[] + files[1:117] + files[118:133] + files[231:243]
        data_label = files[243]
        for j in range(0, len(data_input[0]), 8):
            data_channels = []
            if j>=len(data_input[0]-16):
                break
            else:
                data = data_input[:,j:j+16]#每个样本要处理的数据
                label_temp = data_label[j:j+16]
            '''如果label无效，该样本也放弃'''
            if np.sum(label_temp == 0) == len(label_temp):
                continue
            '''设立label，统计四种类别出现的次数，最多的为该样本label'''
            label = [0]*4
            sum1 = np.sum(label_temp == 1)
            sum2 = np.sum(label_temp == 2)
            sum3 = np.sum(label_temp == 4)
            sum4 = np.sum(label_temp == 5)
            if (sum1>sum2)&(sum1>sum3)&(sum1>sum4):
                label[0] = 1
            elif (sum2>sum1)&(sum2>sum3)&(sum2>sum4):
                label[1] = 1
            elif (sum3>sum1)&(sum3>sum2)&(sum3>sum4):
                label[2] = 1
            elif (sum4>sum1)&(sum4>sum2)&(sum4>sum3):
                label[3] = 1
            else:#不属于任何一个类别，直接舍弃掉当前样本
                label.clear()
                continue

            for chans in data:
                data_temp = []
                if (np.isnan(chans).sum()>= len(chans)//2): #如果nan值数量超过一半以上，该样本作废
                    label.clear()
                    data_channels.clear()
                    break
                else:
                    '''先将值为nan的地方找出来，并将其替换为前一个值'''
                    for k in range(len(chans)):
                        if np.isnan(chans[k]):
                            if k!= 0:
                                chans[k] = chans[k-1]
                            else:
                                chans[k] = 0
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

                    '''数据归一化'''
                    min_data = np.min(data_temp)
                    max_data = np.max(data_temp)
                    for x in range(len(data_temp)):
                        data_temp[x] = (data_temp[x] - min_data)/(max_data - min_data)
                    data_channels.append(np.array(data_temp, dtype = float))
            if label!= []:
                labels.append(np.array(label, dtype = float))
            if data_channels!= []:
                train_data.append(np.array(data_channels, dtype = float))
            print("one sample finish", j//8)
        print("one man finish", i)
    train_data = np.array(train_data, dtype = float, ndmin = 3)
    labels = np.array(labels, dtype = float, ndmin = 2)
    print(train_data.shape) #(22527,143,11)
    print(labels.shape) #(22527,143,4)
    return train_data,labels


path = './data/*'
f_list = glob.glob(path)
train_data, train_label = split_data1(f_list)
np.savez("oppo.npz", data = train_data, label = train_label)
