 function fitvalue= fitness(trandata,D)



datalabel=trandata(:,end);
data=trandata(:,1:end-1);
sfsize=size(trandata,2)-1;
pf=sfsize/D;
% miu=0.533;  %0.835 %0.855
% w=0.01;

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
balance_err=1-result/ctype;

flength=size(trandata,2)-1;
dlength=size(trandata,1);
DB=0;
DW=0;
for i=1:size(trandata,1)
    dindex=find(datalabel~=datalabel(i,:));
    sindex=find(datalabel==datalabel(i,:));
    DBM=[];
    DWM=[];
    for j=1:size(dindex)
        DBM=[DBM size(find((data(i,:)==data(dindex(j,:),:))==1),2)/flength];
    end
    for j=1:size(sindex)
        if i~=sindex(j)
         DWM=[DWM size(find((data(i,:)==data(sindex(j,:),:))==1),2)/flength];  
        end
        if isempty(DWM)  %此处暂时发现9_Tumor中的问题，因为一个类别只有一个，
                            %无法判定距离，暂时将其设置为最优，不影响全局
            DWM=0.3;
        end
    end
 
      DB=DB+max(DBM);  %不同类别之间的最大重合比例
      DW=DW+min(DWM);  %相同类别之间的最小重合比例
end
DB=DB/dlength;
DW=DW/dlength;
distance=1/(1+exp(-5*(DB-DW)));
fitvalue=[balance_err;distance;pf];



