import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from model import Attention_group, Learning_group, Classifier, CSmodule, multimetrics

torch.manual_seed(1)
torch.cuda.manual_seed(1)
batch_train = 100
start_epoch = 0
end_epoch = 40
cs_epoch = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_file_name = './log.txt'
train_data_name = './data/SDA.npz'
data_file = np.load(train_data_name, allow_pickle = True)
train_data, train_label = data_file["train_data"], data_file["train_labels"]
class_data, class_label = train_data, train_label
test_data, test_label = data_file["test_data"], data_file["test_labels"]

test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
transformer = transforms.Compose([transforms.ToTensor()])
testloader = DataLoader(dataset=test_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
criterion = nn.CrossEntropyLoss()


val_loss_list = []
A_weight = 0
A_list = []

K_cnt = 0 
for k_idx in range(8):
    train_loss_list = []
    AFS_atten = Attention_group()
    AFS_learn = Learning_group()
    AFS_cs = CSmodule()
    AFS_atten = AFS_atten.to(device)
    AFS_learn = AFS_learn.to(device)
    AFS_cs = AFS_cs.to(device)
    optimizer = torch.optim.SGD(
        [{'params': AFS_atten.parameters(), 'initial_lr': 0.12}, {'params': AFS_learn.parameters(), 'initial_lr': 0.12}],
        lr=0.12, momentum=0.8, dampening=0.4, weight_decay=0.0001)
    optimizer_cs = torch.optim.Adam(params = AFS_cs.parameters(), lr = 0.1, weight_decay = 0.0001)
    # optimizer = torch.optim.ASGD(
    #     [{'params': AFS_atten.parameters(), 'initial_lr': 0.15}, {'params': AFS_learn.parameters(), 'initial_lr': 0.15}],
    #     lr=0.15, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=start_epoch)
    transformer = transforms.Compose([transforms.ToTensor()])
    train_idx = []
    train_idx = train_idx + [i for i in range(0, k_idx*912)] + [i for i in range((k_idx+1)*912, 7296)]

    train_dataset = TensorDataset(torch.tensor(train_data[train_idx,:,:]), torch.tensor(train_label[train_idx,:]))
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    transformer = transforms.Compose([transforms.ToTensor()])
    val_dataset = TensorDataset(torch.tensor(train_data[int(k_idx*912):int((k_idx+1)*912),:,:]), torch.tensor(train_label[int(k_idx*912):int((k_idx+1)*912),:]))
    valloader = DataLoader(dataset=val_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    criterion1 = nn.MSELoss()
    
    for idx_epoch in range(start_epoch + 1, cs_epoch + 1):
        for idx,(inputs, labels) in enumerate(trainloader):
          inputs = inputs.reshape(int(inputs.shape[0]), 945)
          inputs = inputs.to(device)
          cs_out = AFS_cs(inputs)
          loss = criterion1(inputs.float(), cs_out)
          optimizer_cs.zero_grad()
          loss.backward()
          optimizer_cs.step()
          print('Training in epoch {} cs loss: {}'.format(idx_epoch, loss.item()))
          
    for idx_epoch in range(start_epoch + 1, end_epoch + 1):
        loss_list = []
        A_weight = 0
        total_loss = 0
        '''train all the network to get A'''
        for idx, (inputs, labels) in enumerate(trainloader):
            loss_epoch = 0
            inputs = inputs.to(device)
            label = torch.argmax(labels, 1)
            label = label.to(device)
            
            cs_temp = AFS_cs(inputs.reshape(int(inputs.shape[0]), 945))
            cs_temp = cs_temp.reshape(inputs.shape[0], 45, 21)
            temp, A_weight = AFS_atten(cs_temp)
            outputs = AFS_learn(temp)
            loss = criterion(outputs, label)
            loss_epoch += loss.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('K fold in {}/8 // Training batchsize {} epoch {}/{}  average loss: {}'.format(K_cnt+1, batch_train, idx_epoch, end_epoch, loss_epoch))
        scheduler.step()
        train_loss_list.append(total_loss)
    print('Training total average loss:{}'.format(sum(train_loss_list)/len(train_loss_list)))

    '''val'''
    AFS_atten.eval()
    AFS_learn.eval()
    valloss_epoch = 0
    for idx, (inputs, labels) in enumerate(valloader):
        inputs = inputs.to(device)
        label = torch.argmax(labels, 1)
        label = label.to(device)
        temp, A_weight = AFS_atten(inputs)
        outputs = AFS_learn(temp)
        loss = criterion(outputs, label)
        valloss_epoch += loss.item()
        print('K fold in {}/8 // Val stage: average loss:{}'.format(K_cnt+1, loss.item()))
    A_list.append(A_weight)
    val_loss_list.append(valloss_epoch)
    torch.save(AFS_atten.state_dict(), "./model/AFS_group_atten_net{}.pkl".format(K_cnt))
    torch.save(AFS_learn.state_dict(), "./model/AFS_group_learn_net{}.pkl".format(K_cnt))
    K_cnt += 1



AFS_atten = Attention_group()
AFS_learn = Learning_group()
AFS_atten = AFS_atten.to(device)
AFS_learn = AFS_learn.to(device)
model_idx = val_loss_list.index(min(val_loss_list))
AFS_atten.load_state_dict(torch.load('./model/AFS_group_atten_net{}.pkl'.format(model_idx)))
AFS_learn.load_state_dict(torch.load('./model/AFS_group_learn_net{}.pkl'.format(model_idx)))
A_weight = A_list[model_idx]


'''training classifier stage'''
atten_weight = A_weight.cpu().detach().numpy()
atten_weight = np.mean(atten_weight, 0)
AFS_weight_rank = list(np.argsort(atten_weight))[::-1]
output_file = open(log_file_name, 'a')

for K in range(5, 645, 20):
    total = 0

    AFS_class = Classifier(input_size = K)
    AFS_class = AFS_class.to(device)
    optimizer1 = torch.optim.ASGD([{'params': AFS_class.parameters(), 'initial_lr': 0.12}], lr=0.12,
                                  weight_decay=0.0001)
    # optimizer1 = torch.optim.SGD([{'params': AFS_class.parameters(), 'initial_lr': 0.15}], lr=0.15,
    #                               weight_decay=0.0001)
    train_dataset = TensorDataset(torch.tensor(class_data), torch.tensor(class_label))
    transformer = transforms.Compose([transforms.ToTensor()])
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_train, num_workers=0, shuffle=True)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=start_epoch)
    for temp_epoch in range(start_epoch + 1, end_epoch + 1):
        AFS_class.train()
        '''train classifier to test'''
        for data in trainloader:
            train_inputs, train_labels = data
            l_epoch = 0
            train_inputs = train_inputs.reshape(int(train_inputs.shape[0]), 945)
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
        scheduler1.step()


    pred_y = []
    true_y = []
    AFS_class.eval()
    '''test accuracy'''
    for data in testloader:
        test_inputs, test_labels = data
        test_labels = test_labels.to(device)
        test_inputs = test_inputs.reshape(int(test_inputs.shape[0]), 945)
        test_input = test_inputs[:, AFS_weight_rank[:K]]
        test_input = test_input.to(device)
        test_output = AFS_class(test_input)
        pred = torch.argmax(test_output, 1)
        test_label = torch.argmax(test_labels, 1)
        # acc += torch.eq(pred, test_label).sum().float().item()
        pred_y.extend(list(pred.cpu().numpy()))
        true_y.extend(list(test_label.cpu().numpy()))
    pred_y = np.array(pred_y)
    true_y = np.array(true_y)
    f1,acc,auc = multimetrics(true_y, pred_y)
    #accuracy = acc*100 / len(test_data)
    print("HAP-DNN Using top {} features/ accuracy:{:.4f}% /f1 score:{:.4f} /AUC:{:.4f}".format(K, acc*100, f1, auc))
    output_file.write("HAP-DNN Using top {} features/accuracy:{:.4f}% /f1 score:{:.4f} /AUC:{:.4f}".format(K, acc*100, f1, auc) + '\n')

output_file.write('\n')
output_file.close()

