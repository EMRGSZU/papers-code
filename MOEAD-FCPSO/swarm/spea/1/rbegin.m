function [leng,err]=rbegin(data,Indices,EP)
% ���ܣ���ʼ��������ѡ����������
% ���룺data�������ݼ�����Ϊʵ������Ϊ���ԡ�
% �����classnum�������ǩ������
%       classlabel�������ǩ
%

%����Ԥ�������ǩ��Ϊ���һ��
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
% x=[];
%�����ݼ�����10��cv
% Indices =crossvalind('Kfold',datalabel, 10);
for i=1:10
    disp(i)
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
cut=disc_MDLP(traind);
    for j=1:size(EP{1,i},1)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %��������ѡ����
    x=EP{1,i}(j).Position();
    [leng(i,j),err(i,j)] = testerror(x,traind,testd,cut); %�鿴ѡ�����������ͷ�����ȷ��
    end
end
save matlab1 leng ;
save matlab2 err ;

