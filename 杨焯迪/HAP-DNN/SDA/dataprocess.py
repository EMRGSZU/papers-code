import numpy as np
import os
import glob
import scipy.stats as st
import statsmodels.tsa.api as smt

def show_files(path, all_files):
    path = glob.glob(path + '/*')
    for path_idx in path:
        idx_list = glob.glob(path_idx + '/*') #p01~p08
        for idx in idx_list:
            temp_file = glob.glob(idx + '/*')
            for i in range(len(temp_file)):
                all_files.append(temp_file[i])
    # file_list = os.listdir(path)
    # for file in file_list:
    #     cur_path = os.path.join(path, file)
    #     if os.path.isdir(cur_path):
    #         show_files(cur_path, all_files)
    #     else:
    #         all_files.append(file)
    return all_files

def split_data(file_list): #len(file)=9120=480*19
    labels = []
    train_data = []
    label_idx = 0
    for i in range(len(file_list)):
        if (i%480==0)&(i!=0):
            label_idx += 1
        label_temp = [0]*19
        label_temp[label_idx] = 1

        files = np.loadtxt(file_list[i], dtype = float, delimiter = ',', unpack = True)
        data_channels = []
        for file in files:
            data_temp = []
            data_temp.append(np.min(file))
            data_temp.append(np.max(file))
            data_temp.append(np.mean(file))
            data_temp.append(st.skew(file))
            data_temp.append(st.kurtosis(file))

            fftlist = np.abs(np.fft.fft(file, 25))
            fft_rank = np.argsort(fftlist)[:-1]
            fft_peak = fftlist[fft_rank[:5]]
            for idx in fft_peak:
                data_temp.append(idx)

            crr = smt.stattools.acf(file, nlags=11, fft=False)
            for idx in crr[1:]:
                data_temp.append(idx)
 
            min_data = np.min(data_temp)
            max_data = np.max(data_temp)
            for x in range(len(data_temp)):
                data_temp[x] = (data_temp[x] - min_data)/(max_data - min_data)

            data_channels.append(np.array(data_temp, dtype = float))
        train_data.append(np.array(data_channels, dtype = float))
        labels.append(np.array(label_temp, dtype = float))
        print("one slice finish: ", i)


    train_data = np.array(train_data, dtype = float, ndmin = 3)
    labels = np.array(labels, dtype = float, ndmin = 2)
    data_idx = np.random.permutation(train_data.shape[0])
    shuffle_data = train_data[data_idx,:,:]
    shuffle_labels = labels[data_idx,:]
    print(train_data.shape)
    print(labels.shape)
    return shuffle_data, shuffle_labels

file_path = './data'
file_list = show_files(file_path,[])
train_data, train_labels = split_data(file_list)
np.savez("SDA.npz", data = train_data, labels = train_labels)

