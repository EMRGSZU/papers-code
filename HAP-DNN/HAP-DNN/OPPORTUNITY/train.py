import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from model import Attention_group, Learning_group, Classifier, CSmodule, multimetrics

torch.manual_seed(1)
torch.cuda.manual_seed(1)
batch_train = 100
start_epoch = 0
end_epoch = 40
cs_epoch = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_file_name = './log.txt'
train_data_name = './oppo.npz'
data_file = np.load(train_data_name, allow_pickle = True)
datfile,labfile = data_file['data'],data_file['label']
acc_list = []
f1_list = []
auc_list =[]
data_cnt = datfile.shape[0]
for i in range(1):
    data_idx = np.random.permutation(data_cnt)
    data = np.array(datfile, copy = True)
    label = np.array(labfile, copy = True)
    data = data[data_idx,:,:]
    label = label[data_idx,:]
    train_data, train_label = data[0:int(0.8*data.shape[0]),:,:], label[0:int(0.8*data.shape[0]),:]
    test_data, test_label = data[int(0.8*data.shape[0]):,:,:], label[int(0.8*data.shape[0]):,:]
    transformer = transforms.Compose([transforms.ToTensor()])
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    AFS_atten = Attention_group()
    AFS_learn = Learning_group()
    AFS_cs = CSmodule()
    AFS_atten = AFS_atten.to(device)
    AFS_learn = AFS_learn.to(device)
    AFS_cs = AFS_cs.to(device)
    optimizer = torch.optim.SGD([{'params': AFS_atten.parameters(), 'initial_lr': 0.1}, {'params': AFS_learn.parameters(), 'initial_lr': 0.1}], lr=0.1, momentum=0.8, dampening=0.4, weight_decay=0.0001)
    optimizer_cs = torch.optim.Adam(params=AFS_cs.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=start_epoch)
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.MSELoss()
    '''training CS module'''
    for idx_epoch in range(start_epoch + 1, cs_epoch + 1):
        for idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.reshape(int(inputs.shape[0]), 3003)
            inputs = inputs.to(device)
            cs_out = AFS_cs(inputs)
            loss = criterion1(inputs.float(), cs_out)
            optimizer_cs.zero_grad()
            loss.backward()
            optimizer_cs.step()
            print('RRSS {} Training in epoch {} cs loss: {}'.format(i+1, idx_epoch, loss.item()))

    for idx_epoch in range(start_epoch + 1, end_epoch + 1):
        loss_list = []
        A_weight = 0
        '''train all the network to get A'''
        for idx, (inputs, labels) in enumerate(trainloader):
            loss_epoch = 0
            inputs = inputs.to(device)
            label = torch.argmax(labels, 1)
            label = label.to(device)
            cs_temp = AFS_cs(inputs.reshape(int(inputs.shape[0]), 3003))
            cs_temp = cs_temp.reshape(inputs.shape[0], 143, 21)
            temp, A_weight = AFS_atten(cs_temp)
            outputs = AFS_learn(temp)
            loss = criterion(outputs, label)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('RRSS {}// Training batchsize {} epoch {}/{}  average loss: {}'.format(i+1, batch_train, idx_epoch, end_epoch, loss_epoch))
        scheduler.step()
    '''training classifier stage'''
    atten_weight = A_weight.cpu().detach().numpy()
    atten_weight = np.mean(atten_weight, 0)
    AFS_weight_rank = list(np.argsort(atten_weight))[::-1]

    acc_temp = []
    f1_temp = []
    auc_temp = []
    for K in range(5, 1500, 50):
        total = 0
        acc = 0
        AFS_class = Classifier(input_size=K)
        AFS_class = AFS_class.to(device)
        optimizer1 = torch.optim.ASGD([{'params': AFS_class.parameters(), 'initial_lr': 0.1}], lr=0.1, weight_decay=0.0001)
        for temp_epoch in range(start_epoch + 1, end_epoch + 1):
            AFS_class.train()
            '''train classifier to test'''
            for data in trainloader:
                train_inputs, train_labels = data
                l_epoch = 0
                train_inputs = train_inputs.reshape(int(train_inputs.shape[0]), 3003)
                tr_input = train_inputs[:, AFS_weight_rank[:K]]
                tr_input = tr_input.to(device)
                train_label = torch.argmax(train_labels, 1)
                train_label = train_label.to(device)
                outputs = AFS_class(tr_input)
                loss1 = criterion(outputs, train_label)
                l_epoch += loss1.item()
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                
                print('select {} classifier training: average loss: {}'.format(K, l_epoch))
            scheduler.step()
        pred_y = []
        true_y = []
        AFS_class.eval()
        '''test accuracy'''
        for data in testloader:
            test_inputs, test_labels = data
            test_labels = test_labels.to(device)

            test_inputs = test_inputs.reshape(int(test_inputs.shape[0]), 3003)
            test_input = test_inputs[:, AFS_weight_rank[:K]]
            test_input = test_input.to(device)
            test_output = AFS_class(test_input)
            pred = torch.argmax(test_output, 1)
            test_label = torch.argmax(test_labels, 1)
            pred_y.extend(list(pred.cpu().numpy()))
            true_y.extend(list(test_label.cpu().numpy()))
        pred_y = np.array(pred_y)
        true_y = np.array(true_y)
        f1, acc, auc = multimetrics(true_y, pred_y)
        acc_temp.append(acc)
        f1_temp.append(f1)
        auc_temp.append(auc)
        for l in range(4):
            print("HAP-DNN Using top {} features/ accuracy:{:.4f}% /f1 score:{:.4f} /AUC:{:.4f}".format(K, acc[l]*100, f1[l], auc[l]))
    acc_list.append(np.array(acc_temp, dtype = float))
    f1_list.append(np.array(f1_temp, dtype = float))
    auc_list.append(np.array(auc_temp, dtype = float))

acc_avg = np.average(acc_list, axis = 0)
f1_avg = np.average(f1_list, axis = 0)
auc_avg = np.average(auc_list, axis = 0)
output_file = open(log_file_name, 'a')

for i_idx in range(len(acc_avg)):
    for j_idx in range(len(acc_avg[0])):
        print("HAP-DNN Using top {} features subject{}/ accuracy:{:.4f}% /f1 score:{:.4f} /AUC:{:.4f}".format(i_idx*50+5,j_idx, acc_avg[i_idx,j_idx]*100, f1_avg[i_idx,j_idx], auc_avg[i_idx,j_idx]))
        output_file.write("HAP-DNN Using top {} features subject{}/ accuracy:{:.4f}% /f1 score:{:.4f} /AUC:{:.4f}".format(i_idx*50+5,j_idx, acc_avg[i_idx,j_idx]*100, f1_avg[i_idx,j_idx], auc_avg[i_idx,j_idx]) + '\n')
    output_file.write('\n')
output_file.write('\n')
output_file.close()


