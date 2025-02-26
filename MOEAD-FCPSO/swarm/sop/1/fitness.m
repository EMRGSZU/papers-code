function [fit,err,distance,pf]= fitness(trandata,D,sindex,dindex)
%UNTITLED ��fitnessֵ
%   trandataΪ����ѡ���δ��ɢ����ݼ���DΪ����ά��

datalabel=trandata(:,end);
data=trandata(:,1:end-1);
sfsize=size(trandata,2)-1;
pf=sfsize/D;
%miu=0.543;  %0.835 %0.855
w=0.54; %��?��?0.51??��??��????????
u=0.01;


euDis=pdist(data, 'hamming'); %hamminfֵ���ص�����
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
result=a./b;  %���ƽ����ȷ��?
result=sum(result);
%balance_acc=result/ctype;
err=result/ctype;


%flength=size(trandata,2)-1;
dlength=size(trandata,1);

 ES=zeros(1,dlength);  %hamming same
 ED=zeros(1,dlength);  %hamming different

parfor i=1:dlength
     select_eudis=eudis(i,:);
           dif_eudis=min(select_eudis(dindex{i}));    %DB ��ͬ����֮��ľ���  ��������ԽС��ʾ�����ԽС  ���������ҵ���С�Ľ���Ϊ1 
           same_eudis=max(select_eudis(sindex{i}));
            if isempty(same_eudis)
                same_eudis=0;
             end
           ES(i)= same_eudis;  %��֮ͬ��hammingԽСԽ��
           ED(i)=dif_eudis;  %��֮ͬ��hammingԽ��Խ��
    
end  

ES=sum(ES)/dlength;
ED=sum(ED)/dlength;
distance=1/(1+exp(-5*(ES-ED))); 

fit=w*err+(1-w-u)*distance+u*pf;
%fit=(miu*balance_acc+(1-miu)*distance)-0.001*pf;
end

function pre_class= eu_KNN(test,eudis,datalabel)
    %global datalabel
    sele_test=find(test);    %��ѡ���test��ǩ
    test_size=size(sele_test,1); %��ѡ��test����
    pre_class=zeros(test_size,1);
    classdis=eudis;
    parfor i=1:test_size
    sele_data=sele_test(i);  %��ѡ�����ǵڼ�������
    select_dis=classdis(sele_data,:);
    select_dis(test)=Inf;  %������������������ľ�������Ϊ���޴�
    
    [~,pre_index]=min(select_dis);  %�ҵ��뱾����֮�����С������±�
     pre_class(i)=datalabel(pre_index);   %ͨ��С���ҵ�Ԥ��ı�ǩ
 
  
    end
     
 end



