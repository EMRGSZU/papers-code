function begin(data) 
% ���ܣ���ʼ��������ѡ����������
% ���룺data�������ݼ�����Ϊʵ������Ϊ���ԡ�
% �����classnum�������ǩ������ 
%       classlabel�������ǩ % 
clc; 
clear; 
close all; 
warning('off'); 
dataNameArray={'RELATHE','PCMAC','BASEHOCK'};  
for it=1:size(dataNameArray,2)                    
    clc                    
    clearvars -except dataNameArray it               
    dataname=dataNameArray{it};                
    file=[dataname,'.mat'];                
    load(file);
% diary 'parallel_SRBCT.txt';
%����Ԥ�������ǩ��Ϊ���һ��
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
total_time1 = clock;

M_trianaccall=[];
M_Itertimeall=[];
M_fittimeall=[];
M_rateall=[];
M_uptimeall=[];
M_dotimeall=[];
t_max=10;
for k=1:t_max
 disp(k)
EP=[];
PB=[];
itertime=[];
fittime=[];
rate=[];
uptime=[];
dotime=[];

%�����ݼ�����10��cv
Indices =crossvalind('Kfold',datalabel, 10);

cut=cell(1,10);
parfor i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
cut{i}=disc_MDLP(traind);
end

for i=1:10
itime1=clock;
test=(Indices==i);
train=~test;
traind=data(train,:);

[EP{i},PB{i},itertime(i,:),fittime(i,:),uptime(i,:),rate(i,:),dotime(i,:)]=PotentialPso(150,traind,cut{1,i}); %��������ѡ����
itime2=clock;
PB{1, i} = rmfield(PB{1, i}, 'select');
EP{1, i} = rmfield(EP{1, i}, 'select');
logger(['## fold ', num2str(i), '/', num2str(k), ', time = ', num2str(etime(itime2,itime1)), 's']);
end

M_Itertimeall=[M_Itertimeall;sum(itertime)./10];
M_fittimeall=[M_fittimeall,;sum(fittime)./10];
M_uptimeall=[M_uptimeall;sum(uptime)./10];
M_rateall=[M_rateall;sum(rate)./10];
M_dotimeall=[M_dotimeall;sum(dotime)./10];

indi=strcat('indi',num2str(k),'.mat');
EPI=strcat('EP',num2str(k),'.mat');
cuti=strcat('cut',num2str(k),'.mat');
pbi=strcat('pb',num2str(k),'.mat');

sdataname=[dataname '/']; 
save([sdataname,cuti], 'cut'); 
save([sdataname,indi], 'Indices'); 
save([sdataname,EPI],'EP'); 
save([sdataname,pbi],'PB');
logger(['### round', num2str(k), ' complete']);
end


if t_max==1
    mtimeall=M_Itertimeall;
    mfittimeall=M_fittimeall;
    muptimeall=M_uptimeall;
    mrateall=M_rateall;
    mdotimeall=M_dotimeall;
else
    mtimeall=sum(M_Itertimeall,1)/t_max;
    mfittimeall=sum(M_fittimeall,1)/t_max;
    muptimeall=sum(M_uptimeall,1)/t_max;
    mrateall=sum(M_rateall,1)/t_max;
    mdotimeall=sum(M_dotimeall,1)/t_max;
end

m_itertime=mean(mtimeall);
m_fittime=mean(mfittimeall);
m_uptime=mean(muptimeall);
m_dotime=mean(mdotimeall);

logger(['average iter time = ',num2str(m_itertime)]);
% logger(['average training accuracy = ', num2str(m_train_acc)]);
% logger(['average testing accuracy = ', num2str(m_test_acc)]);
% logger(['std testing accuracy = ', num2str(std_test_acc)]);
% logger(['average feature size = ', num2str(m_feature_size)]); 
total_time2 = clock;
total_time = etime(total_time2, total_time1)
savename=[ 'test2' dataname '.mat' ];  
save(savename);
diary off;
end
end


