function Bbegin(data)
% ���ܣ���ʼ��������ѡ����������
% ���룺data�������ݼ�����Ϊʵ������Ϊ���ԡ�
% �����classnum�������ǩ������
%       classlabel�������ǩ

%����Ԥ�������ǩ��Ϊ���һ��
D=size(data,2)-1;
data=[data(:,2:end) data(:,1)];
data(:,1:end-1)=mapminmax(data(:,1:end-1)',0,1)';
datalabel=data(:,end);
%x=zeros(10,D);
%�����ݼ�����10��cv
Indices =crossvalind('Kfold',datalabel, 10);
cutx=cell(1,10);
for i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
% cut = MDLP(traind);
% traind = dicdata(traind,cut);
% cutx{i}=cut;
EP{i}=Pso(300,traind);
end



