function begin() 
clc; 
clear; 
close all; 
% dataset {'SRBCT','9_Tumors','11_Tumors','Adenocarcinoma','Brain_Tumor1','Brain_Tumor2','Breast3','DLBCL','Leukemia1','Leukemia2','Lung_Cancer','Lymphoma','Nci','Prostate6033','Prostate_Tumor','Brain_Tumor1','Brain_Tumor2','Breast3'}  dataNameArray={'SRBCT'}; 
%'RELATHE','PCMAC','BASEHOCK' 
dataNameArray={'RELATHE','PCMAC','BASEHOCK'}; 
for it=1:size(dataNameArray,2)          
    clc          
    clearvars -except dataNameArray it     
    dataname=dataNameArray{it};      
    file=[dataname,'.mat'];      
    load(file);

% diary 'parallel_SRBCT.txt';
%function begin(data)
% 功能：开始进行特征选择，主函数！
% 输入：data——数据集，行为实例，列为属性。
% 输出：classnum——类标签的数量
%       classlabel——类标签
%

%数据预处理将类标签变为最后一列
 t1=clock;
 
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);

Result_memberall=[];
M_trianaccall=[];
M_Itertimeall=[];
M_fittimeall=[];
M_rateall=[];
M_uptimeall=[];
M_featuresizeall=[];
M_testaccall=[];



%将数据集进行10倍cv
k=10;
t_max=1;
for t=1:t_max
Indices =crossvalind('Kfold',datalabel, k);
cut=cell(1,10);

x=[];
itertime=[];
trianacc=[];
fittime=[];
rate=[];
uptime=[];
error=[];
length=[];

parfor i=1:k
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
cut{i}=disc_MDLP(traind);
end

for i=1:k
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
%cut=disc_MDLP(traind);
[x(i,:),itertime(i,:),trianacc(i),fittime(i,:),uptime(i,:),rate(i,:)]=PotentialPso(150,traind,cut{1,i}); %返回特征选择结果
[length(i,:),error(i,:)] = testerror(x(i,:),traind,testd,cut{1,i}) ;%查看选择特征数量和分类正确率
logger(['## fold ', num2str(i), '/', num2str(k), ', time = ', num2str(itertime(i)), 's']);
end
Result_memberall=[Result_memberall;x];
M_Itertimeall=[M_Itertimeall;sum(itertime)./k];
M_fittimeall=[M_fittimeall,;sum(fittime)./k];
M_uptimeall=[M_uptimeall;sum(uptime)./k];
M_rateall=[M_rateall;sum(rate)./k];
M_trianaccall=[M_trianaccall,mean(trianacc)];
M_testaccall=[M_testaccall,mean(error)];
M_featuresizeall=[M_featuresizeall,mean(length)];

t2=clock;
etime(t2,t1);
end
if t_max==1
    mtimeall=M_Itertimeall;
    mfittimeall=M_fittimeall;
    muptimeall=M_uptimeall;
    mrateall=M_rateall;
else
    mtimeall=sum(M_Itertimeall,1)/t_max;
    mfittimeall=sum(M_fittimeall,1)/t_max;
    muptimeall=sum(M_uptimeall,1)/t_max;
    mrateall=sum(M_rateall,1)/t_max;
end

m_itertime=mean(mtimeall);
m_fittime=mean(mfittimeall);
m_uptime=mean(muptimeall);
m_train_acc=mean(M_trianaccall);
m_test_acc=mean(M_testaccall);
m_feature_size=mean(M_featuresizeall);

logger(['average iter time = ',num2str(m_itertime)]);
logger(['average training accuracy = ', num2str(m_train_acc)]);
logger(['average testing accuracy = ', num2str(m_test_acc)]);
% logger(['std testing accuracy = ', num2str(std_test_acc)]);
logger(['average feature size = ', num2str(m_feature_size)]); 
savename=[ 'test2' dataname '.mat' ];  
save(savename);
diary off;
% save xs x;
% save lens length; %保存文件以便查看结果
% save erroes error;
end
end
