import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skomodel import Attention_group, Learning_group, multimetrics, classification,plot_confusion_matrix,plot_feature
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math


torch.manual_seed(1)
torch.cuda.manual_seed(1)
batch_train = 64
start_epoch = 0
end_epoch = 40

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_file_name = './log.txt'
train_data_name = './data/right_skoda1.npz'
data_file = np.load(train_data_name, allow_pickle=True)
datfile, labfile = data_file['data'], data_file['label']
data_cnt = datfile.shape[0]
acc_list = []
f1_list = []
auc_list =[]
for i in range(1):
    data_idx = np.random.permutation(data_cnt)
    data = np.array(datfile, copy=True)
    label = np.array(labfile, copy=True)
    data = data[data_idx, :, :]
    label = label[data_idx, :]
    train_data, test_data = data[0:int(0.8 * data.shape[0]), :, :], data[int(0.8 * data.shape[0]):, :, :]
    train_label, test_label = label[0:int(0.8 * data.shape[0]), :], label[int(0.8 * data.shape[0]):, :]
    transformer = transforms.Compose([transforms.ToTensor()])
    # valen label train and test
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_train, num_workers=0, shuffle=True)

    # initialize recurrent group difference output as 1
    group_dif = torch.ones(60)
    group_dif = torch.chunk(group_dif, chunks=60, dim=0)
    AFS_atten = Attention_group(g_weight=group_dif)
    # AFS_learn = Learning_group()
    AFS_cla = classification()
    AFS_atten = AFS_atten.to(device)
    # AFS_learn = AFS_learn.to(device)
    AFS_cla = AFS_cla.to(device)
    para_list = []
    para_list.append({'params': AFS_atten.parameters(), 'initial_lr': 0.03})
    para_list.append({'params': AFS_cla.parameters(), 'initial_lr': 0.03})

    optimizer = torch.optim.SGD(para_list, lr=0.03, momentum=0.8, dampening=0.4, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(para_list, lr=0.1,weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=start_epoch)
    criterion = nn.CrossEntropyLoss()

    for idx_epoch in range(start_epoch + 1, end_epoch + 1):
        loss_list = []
        A_weight = 0
        for idx, (inputs, labels) in enumerate(trainloader):
            loss_epoch = 0
            inputs = inputs.to(device)
            label = torch.argmax(labels, 1)
            label = label.to(device)
            temp, A_weight, group_dif,sen_weight,s_mean,weight_diff = AFS_atten(inputs, group_dif)
            # outputs = AFS_learn(temp)
            outputs = AFS_cla(temp)
            lambda1 = 1e-5 # for test
            lambda2 = 1e-5
            lambda3 = 1e-5
            for x in range(len(s_mean)):
                s_mean[x] = s_mean[x] * math.tanh(100*s_mean[x])
            s_mean = sum(s_mean)

            loss = criterion(outputs, label) + lambda1*s_mean + lambda2*torch.sum((A_weight*torch.tanh(100*A_weight))) + lambda3*torch.mean(weight_diff)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('RRSS {}// Training batchsize {} epoch {}/{}  average loss: {}'.format(i + 1, batch_train,
                                                                                         idx_epoch,
                                                                                         end_epoch, loss_epoch))
        scheduler.step()

    '''training classifier stage'''
    atten_weight = A_weight.cpu().detach().numpy()
    atten_weight = np.mean(atten_weight, 0)
    AFS_weight_rank = list(np.argsort(atten_weight))[::-1]
    output_file = open(log_file_name, 'a')
    output_file.write("left skoda sensor selection{}".format(sen_weight) + '\n')
    acc_temp = []
    f1_temp = []
    auc_temp = []
    AFS_atten.eval()
    # AFS_learn.eval()
    AFS_cla.eval()
    pred_y = []
    true_y = []
    tes_gro_dif = torch.ones(60)
    tes_gro_dif = torch.chunk(tes_gro_dif, chunks=60, dim=0)
    for data in testloader:
        test_inputs, test_labels = data
        test_labels = test_labels.to(device)
        test_inputs = test_inputs.to(device)
        temp, tes_weight, tes_gro_dif,sen_weight,s_mean,weight_diff = AFS_atten(test_inputs, tes_gro_dif)
        # test_out = AFS_learn(temp)
        test_out = AFS_cla(temp)
        prelab = torch.argmax(test_out, 1)
        trulab = torch.argmax(test_labels, 1)
        pred_y.extend(list(prelab.cpu().numpy()))
        true_y.extend(list(trulab.cpu().numpy()))
    pred_y = np.array(pred_y)
    true_y = np.array(true_y)
    f1, acc, auc = multimetrics(true_y, pred_y)
    acc_temp.append(acc)
    f1_temp.append(f1)
    auc_temp.append(auc)
    print("fused_net/right/ accuracy:{:.4f} /f1 score:{:.4f} /AUC:{:.4f}".format(acc, f1, auc))
    acc_list.append(np.array(acc_temp, dtype=float))
    f1_list.append(np.array(f1_temp, dtype=float))
    auc_list.append(np.array(auc_temp, dtype=float))
    cm = confusion_matrix(true_y, pred_y)
    labels_name = ["Write on notepad", "Open hood", "Close hood", "Check gaps on the front door",
                   "Open left front door", "Close left front door", "Close both left door",
                   "Check trunk gaps", "Open and close trunk", "Check steering wheel"]
    # plot_confusion_matrix(cm, labels_name, "Skoda(right)")
    # plt.savefig("rigcm.pdf", bbox_inches="tight")
    # plt.show()
    plot_feature(atten_weight)
    plt.savefig("tanrisko.pdf", bbox_inches="tight")
    plt.show()



acc_avg = np.average(acc_list, axis = 0)
f1_avg = np.average(f1_list, axis = 0)
auc_avg = np.average(auc_list, axis = 0)
output_file = open(log_file_name, 'a')

for i_idx in range(len(acc_avg)):
    print("Fused_net/skoda/ accuracy:{:.4f} /f1 score:{:.4f} /AUC:{:.4f}".format(acc_avg[i_idx], f1_avg[i_idx], auc_avg[i_idx]))
    output_file.write("Fused_net/skoda/ accuracy:{:.4f} /f1 score:{:.4f} /AUC:{:.4f}".format(acc_avg[i_idx], f1_avg[i_idx], auc_avg[i_idx]) + '\n')
output_file.write('\n')
output_file.close()


