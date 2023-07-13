import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import math
size = 6 #sensor axis size
s_num = 10 #sensor number
g_num = 60 #channel number
f_num = 11 #feature number
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Fused(nn.Module):
    def __init__(self):
        super(Fused, self).__init__()
        self.max_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,padding=size,stride= size, bias=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3, stride=3, bias=True),
            nn.Conv2d(in_channels=1, out_channels=size, kernel_size=3, bias=True)
        )
        self.avg_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, padding=size, stride=size, bias=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3, stride=3, bias=True),
            nn.Conv2d(in_channels=1, out_channels=size, kernel_size=3, bias=True)
        )


    def forward(self, g_weight, max_list, avg_list):
        g_avg_matrix = torch.zeros([g_weight.shape[0], size, size])
        g_max_matrix = torch.zeros([g_weight.shape[0], size, size])
        g_diff_matrix = torch.zeros([g_weight.shape[0], size, size])

        g_diff = 0

        for i_idx in range(0,size):
            for j_idx in range(0,size):
                g_avg_matrix[:, i_idx, j_idx] = torch.abs(avg_list[:,j_idx].sub(avg_list[:,i_idx]))
                g_max_matrix[:, i_idx, j_idx] = torch.abs(max_list[:, j_idx].sub(max_list[:, i_idx]))
                g_diff_matrix[:, i_idx, j_idx] = torch.abs(g_weight[:, j_idx].sub(g_weight[:, i_idx]))
                g_temp = torch.abs(g_weight[:, j_idx].sub(g_weight[:, i_idx]))
                g_diff = g_diff + g_temp*torch.tanh(100*g_temp)
        g_diff = g_diff / 2
        g_avg_matrix = self.avg_conv(g_avg_matrix.cuda().unsqueeze(1))
        g_max_matrix = self.max_conv(g_max_matrix.cuda().unsqueeze(1))

        dif_temp_vet = (g_avg_matrix + g_max_matrix).squeeze(2)
        diff_output = torch.bmm(g_diff_matrix.cuda(), dif_temp_vet).squeeze(2)
        return diff_output, g_diff

class  Self_atten(nn.Module):
    def __init__(self):
        super(Self_atten, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        '''self-attention parameter'''
        self.q_conv = nn.Conv1d(in_channels=1, out_channels=f_num*size, kernel_size=1)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=f_num*size, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=f_num*size, kernel_size=1)
        self.o_conv = nn.Conv1d(in_channels=f_num*size, out_channels=1, kernel_size=1)

    def forward(self, x):
        temp_tensor = x.unsqueeze(1)
        proj_q = self.q_conv(temp_tensor).permute(0, 2, 1)
        proj_k = self.k_conv(temp_tensor)
        proj_v = self.v_conv(temp_tensor)
        energy = torch.bmm(proj_q, proj_k)
        attent = self.softmax(energy).permute(0, 2, 1)
        out = torch.bmm(proj_v, attent)
        out = torch.abs(self.o_conv(out))
        return out.squeeze(1)


class Attention_group(nn.Module):
    def __init__(self, g_weight):
        super(Attention_group, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(g_num, g_num//2, bias=True),
            nn.Tanh(),
            nn.Linear(g_num//2, g_num, bias=True),
            nn.Tanh()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=f_num)
        self.avgpool = nn.AvgPool1d(kernel_size=f_num)
        self.group_conv = nn.Conv1d(in_channels=1, out_channels=f_num, kernel_size=f_num)
        self.fused_list = nn.ModuleList([Fused() for i in range(s_num)])


    def forward(self, x, g_weight):
        x = x.float()
        max_list = []
        avg_list = []
        for i in range(x.shape[1]):
            temp = x[:, i, :]
            temp = self.group_conv(temp.unsqueeze(1))
            temp = g_weight[i].float().cuda() * temp
            max_temp = self.maxpool(temp.transpose(2, 1))
            avg_temp = self.avgpool(temp.transpose(2, 1))
            max_list.append(max_temp.squeeze(2))
            avg_list.append(avg_temp.squeeze(2))
        max_list = torch.cat(max_list, dim=1)
        avg_list = torch.cat(avg_list, dim=1)
        group_atten = torch.abs(F.softmax(self.mlp(max_list) + self.mlp(avg_list), dim=1))
        # segment channel weight to get sensor weight
        g_a_unstack = torch.chunk(group_atten, chunks=g_num, dim=1)
        g_weight_unstack = torch.chunk(group_atten, chunks=s_num, dim=1)
        #compute sensor weight
        sensor_weight = []
        sensor_mean = []
        for i in range(0,g_num,size):
            temp = g_a_unstack[i]
            for j in range(1,size):
                 temp = temp + g_a_unstack[i+j]
            sensor_weight.append(temp)
            temp = temp.cpu().detach().numpy()
            temp = np.mean(temp, 0)[0]
            sensor_mean.append(temp)
        # sensor_mean = sensor_mean
        sensor_weight_rank = list(np.argsort(sensor_mean))[::-1]

        '''compute avg_list and max_list'''
        max_list_unstack = torch.chunk(max_list, chunks=s_num, dim=1)
        avg_list_unstack = torch.chunk(avg_list, chunks=s_num, dim=1)
        fused_list = []

        weight_diff = 0
        for sen_idx,l in enumerate(self.fused_list):
            temp1,weight_diff = self.fused_list[sen_idx](g_weight_unstack[sen_idx],max_list_unstack[sen_idx],avg_list_unstack[sen_idx])
            fused_list.append(temp1)
            # fused_list.append(self.fused_list[sen_idx](g_weight_unstack[sen_idx],max_list_unstack[sen_idx],avg_list_unstack[sen_idx]))

        fused_list = torch.cat(fused_list, dim=1)
        fused_out = fused_list.cpu().detach().numpy()
        fused_out_mean = np.mean(fused_out, 0)
        fused_out_unstack = torch.chunk(torch.tensor(fused_out_mean), chunks=g_num, dim=0)

        A = torch.ones_like(x)
        Self_net = []
        #test use single self atten net
        for i in range(s_num//2):
            Self_net.append(Self_atten())
        for i in range(s_num // 2):
            Self_net[i] = Self_net[i].to(device)
        self_cnt = 0
        for j in range(s_num):
            if j in sensor_weight_rank[:s_num//2]:
                x_list = []
                for x_idx in range(size):
                    x_list.append(x[:,j*size+x_idx,:])
                x_list = torch.cat(x_list, dim=1)
                self_out = Self_net[self_cnt](x_list)*sensor_weight[j]
                self_out = self_out.reshape(self_out.shape[0], 6, self_out.shape[1] // 6)

                self_cnt = self_cnt+1
                for x_idx in range(size):
                    A[:,j*size+x_idx,:] = self_out[:,x_idx,:]
            else:
                for x_idx in range(size):
                    atten_temp = torch.zeros_like(x[:,j,:])
                    A[:,j*size+x_idx,:] = atten_temp
        x = x.contiguous().view(int(x.shape[0]), g_num*f_num)
        A = A.contiguous().view(int(A.shape[0]), g_num*f_num)
        G = torch.mul(x, A)

        return G,A, fused_out_unstack,sensor_weight_rank,sensor_mean,weight_diff


class Learning_group(nn.Module):
    def __init__(self, input_size=g_num*f_num, L_node=g_num*f_num//2, output_size=10):
        super(Learning_group, self).__init__()
        self.L_node = L_node
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.L_W1 = nn.Linear(input_size, L_node, bias=True)
        self.L_W2 = nn.Linear(L_node, L_node // 2, bias=True)
        self.L_W3 = nn.Linear(L_node // 2, output_size, bias=True)

    def forward(self, x):
        x = x.float()
        L_FC = self.tanh(self.L_W1(x))
        L_FC = self.tanh(self.L_W2(L_FC))
        O = self.L_W3(L_FC)
        return O


def multimetrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    labels = [i for i in range(0, 10)]
    ytrue = label_binarize(y_true, classes=labels)
    ypred = label_binarize(y_pred, classes=labels)
    auc = roc_auc_score(ytrue, ypred, average='weighted')
    return f1, acc, auc

class classification(nn.Module):
    def __init__(self, output_size=10):
        super(classification, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.output_size = output_size
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)
        self.resnet.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=2,padding=3,bias=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], g_num, f_num)
        x = x.unsqueeze(1)
        out = self.resnet(x)
        return out


def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, fontsize=7, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=7, rotation=0)  # 将标签印在y轴坐标上

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                plt.text(j, i, format(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_feature(list_rank):
    # fm = np.zeros((g_num,f_num))
    # for i in range(len(list_rank)//2):
    #     fm[list_rank[i]//11][list_rank[i]%11]=1
    min_data = np.min(list_rank)
    max_data = np.max(list_rank)
    for i in range(list_rank.shape[0]):
        list_rank[i] = (list_rank[i] - min_data)/(max_data-min_data)
    list_rank = list_rank.reshape(g_num,f_num)

    x_name = ["Minimum","Maximum","Mean value","Variance","Skewness","Kurtosis","DFT1","DFT2","DFT3","DFT4","DFT5"]
    y_name = ["1-X calibrated","1-Y calibrated","1-Z calibrated","1-X raw","1-Y raw","1-Z raw",
              "2-X calibrated","2-Y calibrated","2-Z calibrated","2-X raw","2-Y raw","2-Z raw",
              "14-X calibrated","14-Y calibrated","14-Z calibrated","14-X raw","14-Y raw","14-Z raw",
              "16-X calibrated","16-Y calibrated","16-Z calibrated","16-X raw","16-Y raw","16-Z raw",
              "18-X calibrated","18-Y calibrated","18-Z calibrated","18-X raw","18-Y raw","18-Z raw",
              "21-X calibrated","21-Y calibrated","21-Z calibrated","21-X raw","21-Y raw","21-Z raw",
              "22-X calibrated","22-Y calibrated","22-Z calibrated","22-X raw","22-Y raw","22-Z raw",
              "24-X calibrated","24-Y calibrated","24-Z calibrated","24-X raw","24-Y raw","24-Z raw",
              "27-X calibrated","27-Y calibrated","27-Z calibrated","27-X raw","27-Y raw","27-Z raw",
              "29-X calibrated","29-Y calibrated","29-Z calibrated","29-X raw","29-Y raw","29-Z raw"]
    # y_name = ["3-X calibrated", "3-Y calibrated", "3-Z calibrated", "3-X raw", "3-Y raw", "3-Z raw",
    #           "17-X calibrated", "17-Y calibrated", "17-Z calibrated", "17-X raw", "17-Y raw", "17-Z raw",
    #           "19-X calibrated", "19-Y calibrated", "19-Z calibrated", "19-X raw", "19-Y raw", "19-Z raw",
    #           "20-X calibrated", "20-Y calibrated", "20-Z calibrated", "20-X raw", "20-Y raw", "20-Z raw",
    #           "23-X calibrated", "23-Y calibrated", "23-Z calibrated", "23-X raw", "23-Y raw", "23-Z raw",
    #           "25-X calibrated", "25-Y calibrated", "25-Z calibrated", "25-X raw", "25-Y raw", "25-Z raw",
    #           "26-X calibrated", "26-Y calibrated", "26-Z calibrated", "26-X raw", "26-Y raw", "26-Z raw",
    #           "28-X calibrated", "28-Y calibrated", "28-Z calibrated", "28-X raw", "28-Y raw", "28-Z raw",
    #           "30-X calibrated", "30-Y calibrated", "30-Z calibrated", "30-X raw", "30-Y raw", "30-Z raw",
    #           "31-X calibrated", "31-Y calibrated", "31-Z calibrated", "31-X raw", "31-Y raw", "31-Z raw"]
    x_local = np.array(range(len(x_name)))
    y_local = np.array(range(len(y_name)))
    plt.imshow(list_rank,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(x_local, x_name, fontsize=5, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(y_local, y_name, fontsize=5, rotation=0)  # 将标签印在y轴坐标上
    plt.title("Skoda(right)")