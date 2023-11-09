function [balance_acc,fit,distance]= fitness(trandata)
%UNTITLED 求fitness值
%   trandata为特征选择后未离散的数据集，D为特征维度

datalabel=trandata(:,end);
data=trandata(:,1:end-1);
balance_acc=0;
%sfsize=size(trandata,2)-1;
%pf=sfsize/D;
miu=0.543;  %0.835 %0.855

Indices =crossvalind('Kfold',datalabel, 10);
cp=classperf(datalabel);

ctype=size(unique(datalabel(:,end)),1);

for i=1:10  
test=(Indices==i);
train=~test;
train_data=data(train,:);
train_l=datalabel(train,:);
test_data=data(test,:);
test_l=datalabel(test,:);

Mdl=fitcknn(train_data,train_l);
class=predict(Mdl,test_data);
% class = knnclassify(test_data,train_data,train_l);
classperf(cp,class,test);

end
a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=(b-a)./b;  %求解平均正确率
result=sum(result);
balance_acc=result/ctype;

flength=size(trandata,2)-1;
dlength=size(trandata,1);
DBM=[];
DWM=[];

parfor i=1:dlength
    dindex=find(datalabel~=datalabel(i,:));
    sindex=find(datalabel==datalabel(i,:));
    sindex=sindex(sindex~=i);
    DBM=[DBM max(sum(data(i,:)==data(dindex,:),2))/flength];
    DWM=[DWM min(sum(data(i,:)==data(sindex,:),2))/flength];
end  

DB=sum(DBM)/dlength;
DW=sum(DWM)/dlength;

distance=1/(1+exp(-5*(DW-DB))); 
fit=(miu*balance_acc+(1-miu)*distance);



