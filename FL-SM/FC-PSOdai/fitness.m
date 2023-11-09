 function fitvalue= fitness(trandata,D,sindex,dindex)



datalabel=trandata(:,end);
data=trandata(:,1:end-1);
sfsize=size(trandata,2)-1;
if sfsize==0
    balance_err=1;
    distance=1;
    pf=1;

    fitvalue=[balance_err;distance;pf];
    return
end


pf=sfsize/D;
% miu=0.533;  %0.835 %0.855
% w=0.01;
Dis1 = pdist(data, 'hamming');
dis2 = squareform(Dis1);

Indices =crossvalind('Kfold',datalabel, 10);
cp=classperf(datalabel);

ctype=size(unique(datalabel(:,end)),1);

for i=1:10
test=(Indices==i);
train=~test;
train_data=data(train,:);
train_label=datalabel(train,:);
test_data=data(test,:);
test_l=datalabel(test,:);
% class= City_KNN(test,datalabel, dis2);
% class = knnclassify(test_data,train_data,train_l);
Mdl=fitcknn(train_data,train_label);
class=predict(Mdl,test_data);
classperf(cp,class,test);

end
a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=a./b;
result=sum(result);
balance_err=result/ctype;

dlength=size(trandata,1);
 ES=zeros(1,dlength);  %hamming same
 ED=zeros(1,dlength);  %hamming different
for i=1:dlength
    %testlabel=datalabel(i);
    %dindex=find(testlabel~=datalabel);  %��ͬ����
    %sindex=find(testlabel==datalabel); %��ͬ����
    %sindex=sindex(sindex~=i); %��ͬ����,�ų��Լ�
    select_dis=dis2(i,:);
    difdis=min(select_dis(dindex{i}));    %DB ��ͬ����֮��ľ���  ��������ԽС��ʾ�����ԽС  ���������ҵ���С�Ľ���Ϊ1 
   
    samedis=max(select_dis(sindex{i})); %DW ��ͬ����֮��ľ��� ��������Խ�󣬱�ʾ���Խ�� ��������ԽСԽ��  
 
    if isempty(samedis)
        samedis=0;
    end
    

    ES(i)= samedis;  %��֮ͬ��hammingԽСԽ��
      ED(i)=difdis;  %��֮ͬ��hammingԽ��Խ�� 
 
  
    
end    
ES=sum(ES)/dlength;
ED=sum(ED)/dlength;
distance=1/(1+exp(-5*(ES-ED))); 
fitvalue=[balance_err;distance;pf];
 end
 
 
%  function pre_class= City_KNN(test,datalabel, distances)
%     
%     sele_test=find(test);    %��ѡ���test��ǩ
%     test_size=size(sele_test,1); %��ѡ��test����
%     pre_class=zeros(test_size,1);
%     for i=1:test_size
%     sele_data=sele_test(i);  %��ѡ�����ǵڼ�������
%     select_dis=distances(sele_data,:);
%     select_dis(test)=Inf;  %�������뱾�����ľ�������Ϊ���޴�
%     
%     [~,pre_index]=min(select_dis);  %�ҵ��뱾����֮�����С������±�
%     %cand_index=find(select_dis==minvalue);
%     %if size(cand_index,2)==1
%         pre_class(i)=datalabel(pre_index);   %ͨ��С���ҵ�Ԥ��ı�ǩ
%     %else
%      %   cand_class=datalabel(cand_index);
%      %   pre_class(i)=mode(cand_class);
%     %end
%     
%     
%     end
%      
% end



