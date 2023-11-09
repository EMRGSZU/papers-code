import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CSmodule(nn.Module):
    def __init__(self, input_size=264, E_node=26):  # 0.1 compress
        super(CSmodule, self).__init__()
        self.encoder = nn.Linear(input_size, E_node)
        self.decoder = nn.Sequential(
            nn.Linear(E_node, 132, bias=True),
            nn.Tanh(),
            nn.Linear(132, input_size, bias=True),
            nn.Tanh(),
            nn.Linear(input_size, input_size, bias=True),
        )

    def forward(self, x):
        x = x.float()
        E = self.encoder(x)
        E = self.decoder(E)
        return E

class Self_atten(nn.Module):
    def __init__(self):
        super(Self_atten, self).__init__()
        '''self-attention parameter'''
        self.q_conv = nn.Conv1d(in_channels=1, out_channels=21, kernel_size=1)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=21, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=21, kernel_size=1)
        self.o_conv = nn.Conv1d(in_channels=21, out_channels=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        temp_tensor = x.unsqueeze(1)
        proj_q = self.q_conv(temp_tensor).permute(0, 2, 1)
        proj_k = self.k_conv(temp_tensor)
        proj_v = self.v_conv(temp_tensor)
        energy = torch.bmm(proj_q, proj_k)
        attent = self.softmax(energy).permute(0, 2, 1)
        out = torch.bmm(proj_v, attent)
        out = self.o_conv(out)
        return out.squeeze(1)

class Attention_group(nn.Module):
    def __init__(self):
        super(Attention_group, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(24, 12, bias=True),
            nn.Tanh(),
            nn.Linear(12, 24, bias=True),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool1d(kernel_size=11)
        self.avgpool = nn.AvgPool1d(kernel_size=11)
        self.group_conv = nn.Conv1d(in_channels=1, out_channels=11, kernel_size=11)
        '''self-attention parameter'''
        self.q_conv = nn.Conv1d(in_channels=1, out_channels=11, kernel_size=1)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=11, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=11, kernel_size=1)
        self.o_conv = nn.Conv1d(in_channels=11, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = x.float()
        max_list = []
        avg_list = []

        for i in range(x.shape[1]):
            temp = x[:, i, :]  
            temp = self.group_conv(temp.unsqueeze(1))  
            max_temp = self.maxpool(temp.transpose(2, 1))
            avg_temp = self.avgpool(temp.transpose(2, 1))  
            max_list.append(max_temp.squeeze(2))
            avg_list.append(avg_temp.squeeze(2))
        max_list = torch.cat(max_list, dim=1)  
        avg_list = torch.cat(avg_list, dim=1)
        group_atten = F.softmax(self.mlp(max_list) + self.mlp(avg_list), dim=1)  
        group_atten_mean = group_atten.cpu().detach().numpy()
        group_atten_mean = np.mean(group_atten_mean, 0)
        g_a_unstack = torch.chunk(group_atten, chunks=24, dim=1)  
        group_weight_rank = list(np.argsort(group_atten_mean))[::-1]

        A = torch.ones_like(x)
        Self_net = []
        for i in range(24 // 2):
            Self_net.append(Self_atten())
        for i in range(24 // 2):
            Self_net[i] = Self_net[i].to(device)
        c_cnt = 0

        for j in range(x.shape[1]):
            if j in group_weight_rank[:24//2]:
                out = Self_net[c_cnt](x[:, j, :]) * g_a_unstack[j]
                c_cnt += 1
                A[:, j, :] = out
            else:
                atten_temp = torch.zeros_like(x[:, j, :])
                A[:, j, :] = atten_temp
        x = x.contiguous().view(int(x.shape[0]), 264)
        A = A.contiguous().view(int(A.shape[0]), 264)
        G = torch.mul(x, A)
        return G, A


class Learning_group(nn.Module):
    def __init__(self, input_size=264, L_node=132, output_size=13):
        super(Learning_group, self).__init__()
        self.L_node = L_node
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.L_W1 = nn.Linear(input_size, L_node, bias=True)
        self.L_W2 = nn.Linear(L_node, output_size, bias=True)

    def forward(self, x):
        x = x.float()
        L_FC = self.relu(self.L_W1(x))
        O = self.L_W2(L_FC)
        return O


class Classifier(nn.Module):
    def __init__(self, input_size, L_node=132, output_size=13):
        super(Classifier, self).__init__()
        self.L_node = L_node
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.L_W1 = nn.Linear(input_size, self.L_node, bias=True)
        self.L_W2 = nn.Linear(self.L_node, self.output_size, bias=True)

    def forward(self, x):
        x = x.float()
        L_FC = self.relu(self.L_W1(x))
        O = self.L_W2(L_FC)
        return O


def multimetrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    labels = [i for i in range(0, 13)]
    ytrue = label_binarize(y_true, classes=labels)
    ypred = label_binarize(y_pred, classes=labels)
    auc = roc_auc_score(ytrue, ypred, average='weighted')
    return f1, acc, auc
