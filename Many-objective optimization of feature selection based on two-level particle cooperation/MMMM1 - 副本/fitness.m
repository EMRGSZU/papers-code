 function [fitvalue,err,distance,pf]= fitness(trandata,X,D)

cin=find(X);
sfsize=size(cin,2);
pf=sfsize/D;
datalabel=trandata(:,end);
data=trandata(:,cin);
w=0.55; %�ϸ�0.51Ч�����?��

Indices =crossvalind('Kfold',datalabel, 10);
cp=classperf(datalabel);

ctype=size(unique(datalabel(:,end)),1);

for i=1:10
test=(Indices==i);
train=~test;
train_data=data(train,:);
train_l=datalabel(train,:);
test_data=data(test,:);
% test_l=datalabel(test,:);

Mdl=fitcknn(train_data,train_l);
class=predict(Mdl,test_data);
% class = knnclassify(test_data,train_data,train_l);
classperf(cp,class,test);

end
a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=(b-a)./b;
result=sum(result);
err=1-result/ctype;


flength=size(data,2);
dlength=size(data,1);
%DB=0;
%DW=0;
DBM=[];
DWM=[];
parfor i=1:dlength
    dindex=find(datalabel~=datalabel(i,:));
    sindex=find(datalabel==datalabel(i,:));
    sindex=sindex(sindex~=i);
    dl=size(dindex,1);
    sl=size(sindex,1);
    DBM=[DBM min(sum(abs(repmat(data(i,:),dl,1)-data(dindex,:)),2))/flength];
    DWM=[DWM max(sum(abs(repmat(data(i,:),sl,1)-data(sindex,:)),2))/flength];
    %sl=size(sindex,1);   
    %DBM=[DBM min(sum(abs(data(i,:)-data(dindex,:)),2))/flength]; %��ͬ��
   % if sl==0
   %     DWM=[DWM 0.3]
   % else  
    %DWM=[DWM max(sum(abs(data(i,:)-data(sindex,:)),2))/flength]; %��ͬ��
   % end
 
    %  DB=DB+max(DBM);  %��ͬ���֮�������غϱ���
    %  DW=DW+min(DWM);  %��ͬ���֮�����С�غϱ���
end
DB=sum(DBM)/dlength;
DW=sum(DWM)/dlength;
distance=1/(1+exp(-5*(DW-DB)));
fitvalue=[err distance pf];

%fitvalue=w*err+(1-w)*distance;



