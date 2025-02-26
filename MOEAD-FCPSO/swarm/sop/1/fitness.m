function [fit,err,distance,pf]= fitness(trandata,D,sindex,dindex)
%UNTITLED 锟斤拷fitness值
%   trandata为锟斤拷锟斤拷选锟斤拷锟轿达拷锟缴拷锟斤拷锟捷硷拷锟斤拷D为锟斤拷锟斤拷维锟斤拷

datalabel=trandata(:,end);
data=trandata(:,1:end-1);
sfsize=size(trandata,2)-1;
pf=sfsize/D;
%miu=0.543;  %0.835 %0.855
w=0.54; %锟斤拷?锟斤拷?0.51??锟斤拷??锟斤拷????????
u=0.01;


euDis=pdist(data, 'hamming'); %hamminf值非重叠比例
%eudis=squareform(euDis)/sqrt(sfsize);

eudis=squareform(euDis);


Indices =crossvalind('Kfold',datalabel, 10);
cp=classperf(datalabel);

ctype=size(unique(datalabel(:,end)),1);

for i=1:10  
test=(Indices==i);

class= eu_KNN(test,eudis,datalabel);
% class = knnclassify(test_data,train_data,train_l);
classperf(cp,class,test);

end
a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=a./b;  %锟斤拷锟狡斤拷锟斤拷锟饺凤拷锟?
result=sum(result);
%balance_acc=result/ctype;
err=result/ctype;


%flength=size(trandata,2)-1;
dlength=size(trandata,1);

 ES=zeros(1,dlength);  %hamming same
 ED=zeros(1,dlength);  %hamming different

parfor i=1:dlength
     select_eudis=eudis(i,:);
           dif_eudis=min(select_eudis(dindex{i}));    %DB 不同样本之间的距离  汉明距离越小表示，差距越小  本案例中找到最小的将变为1 
           same_eudis=max(select_eudis(sindex{i}));
            if isempty(same_eudis)
                same_eudis=0;
             end
           ES(i)= same_eudis;  %相同之间hamming越小越好
           ED(i)=dif_eudis;  %不同之间hamming越大越好
    
end  

ES=sum(ES)/dlength;
ED=sum(ED)/dlength;
distance=1/(1+exp(-5*(ES-ED))); 

fit=w*err+(1-w-u)*distance+u*pf;
%fit=(miu*balance_acc+(1-miu)*distance)-0.001*pf;
end

function pre_class= eu_KNN(test,eudis,datalabel)
    %global datalabel
    sele_test=find(test);    %所选择的test标签
    test_size=size(sele_test,1); %所选的test个数
    pre_class=zeros(test_size,1);
    classdis=eudis;
    parfor i=1:test_size
    sele_data=sele_test(i);  %所选样本是第几个样本
    select_dis=classdis(sele_data,:);
    select_dis(test)=Inf;  %测试样本与测试样本的距离设置为无限大
    
    [~,pre_index]=min(select_dis);  %找到与本样本之间的最小距离的下表
     pre_class(i)=datalabel(pre_index);   %通过小标找到预测的标签
 
  
    end
     
 end



