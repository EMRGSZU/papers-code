function begin(data)
% ���ܣ���ʼ��������ѡ��������
% ���룺data������ݼ�����Ϊʵ����Ϊ���ԡ�?
% �����classnum�������ǩ������?
%       classlabel��������?
%

%���Ԥ���?���ǩ��Ϊ���һ��
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
D=size(data,2)-1;
M=150;
 %M=round(D/20);
 %if M<150
 %    M=150;
% end
% if M>300
%     M=300;
% end
for k=1:10
disp(k)    
    

%����ݼ�����?10��cv
Indices =crossvalind('Kfold',datalabel, 10);
cut=cell(1,10);
parfor i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
cut{i}=disc_MDLP(traind);
end


for i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
%cut=disc_MDLP(traind);

[PB{i},x(i)]=PotentialPso(M,traind,cut{1,i}); %��������ѡ����
[length(i,:),acc(i,:)] = testerror(x(i).Position,traind,testd,cut{1,i}); %�鿴ѡ�����������ͷ�����ȷ��
end

sum(length)/10
sum(acc)/10
lens=strcat('lens',num2str(k),'.mat');
accuracy=strcat('accuracy',num2str(k),'.mat');
indi=strcat('indi',num2str(k),'.mat');
xi=strcat('xi',num2str(k),'.mat');
pb=strcat('pb',num2str(k),'.mat');


save(['result/',lens],'length');
save(['result/',accuracy],'acc');
save(['result/',indi], 'Indices');
save(['result/',xi],'x');
save(['result/',pb],'PB');

%save xs x;
%save lens length; %�����ļ��Ա�鿴���
%save erroes error;

end

