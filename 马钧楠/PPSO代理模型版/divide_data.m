function divide_data(data)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
datalabel=data(:,end);
data=data(:,1:end-1);
%Indices   =  crossvalind('Kfold', length(datalabel), 10);%10倍cv验证，稍后更改
Indices =crossvalind('Kfold',datalabel, 2);
% for i=1:10
% site = find(Indices~=1);
% train_data = data(site,:);
% train_l=datalabel(site,:);
% site2 = find(Indices==1);
% test_data = data(site2,:);
% test_l=datalabel(site2,:);
% Mdl=fitcknn(train_data,train_l);
% end

% 
% site = find(Indices~=1);
% train_data = data(site,:);
% train_l=datalabel(site,:);
% site2 = find(Indices==1);
% test_data = data(site2,:);
% test_l=datalabel(site2,:);


test=(Indices==1);
train=~test;
train_data=data(train,:);
train_l=datalabel(train,:);
test_data=data(test,:);
test_l=datalabel(test,:);

cp=classperf(datalabel);
class = knnclassify(test_data,train_data,train_l);
classperf(cp,class,test);

% Mdl=fitcknn(train_data,train_l);



% train_data = test(site,:);
% train_l=label(site,:);
% site2 = find(Indices==1);
% test_data = test(site2,:);
% test_l=label(site2,:);
% Mdl=fitcknn(train_data,train_l);
% classlabel=predict(Mdl,test_data);
save classlabel;
%load calsslabel；

%cp中1 n 234  t
% c0=size(find(test_l==0));
% c1=size(find(test_l==1));
% c2=size(find(test_l==2));
% c3=size(find(test_l==3));

% b_acc=

