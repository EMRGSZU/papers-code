function begin(data)
% 功能：开始进行特征选择，主函数！
% 输入：data——数据集，行为实例，列为属性。
% 输出：classnum——类标签的数量
%       classlabel——类标签
%

%数据预处理将类标签变为最后一列
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
total_time1 = clock;
for k=1:10
 disp(k)
EP=[];
PB=[];
% 
% EP3=[];
% PB3=[];
% EP4=[];
% PB4=[];
% EP5=[];
% PB5=[];
% EP6=[];
% PB6=[];

%将数据集进行10倍cv
Indices =crossvalind('Kfold',datalabel, 10);

cut=cell(1,10);
parfor i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
cut{i}=disc_MDLP(traind);
end

for i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
%testd=data(test,:);
%cut=disc_MDLP(traind);
%[EP{i},PB{i},EP3{i},PB3{i},EP4{i},PB4{i},EP5{i},PB5{i},EP6{i},PB6{i}]=PotentialPso(150,traind,cut{1,i});
[EP{i},PB{i}]=PotentialPso(150,traind,cut{1,i}); %返回特征选择结果
% [length(i,:),error(i,:)] = testerror(x(i,:),traind,testd,cut) %查看选择特征数量和分类正确率
%  for j=1:size(EP{1,i},1)
% % [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %返回特征选择结果
%     s=EP{1,i}(j).Position();
%    [leng(i,j),acc(i,j)] = testerror(s,traind,testd,cut{1,i}); %查看选择特征数量和分类正确率
%    
%  end
   %error(i,:)=max(acc(i,:));
   %length(i,:)=min(leng(i,find(acc(i,:)==error(i))));
end
%sum(length)/10
%sum(error)/10

%lens=strcat('lens',num2str(k),'.mat');
%accuracy=strcat('acc',num2str(k),'.mat');
indi=strcat('indi',num2str(k),'.mat');
EPI=strcat('EP',num2str(k),'.mat');
% EPI3=strcat('EP3',num2str(k),'.mat');
% EPI4=strcat('EP4',num2str(k),'.mat');
% EPI5=strcat('EP5',num2str(k),'.mat');
% EPI6=strcat('EP6',num2str(k),'.mat');
cuti=strcat('cut',num2str(k),'.mat');
pbi=strcat('pb',num2str(k),'.mat');
% pbi3=strcat('pb3',num2str(k),'.mat');
% pbi4=strcat('pb4',num2str(k),'.mat');
% pbi5=strcat('pb5',num2str(k),'.mat');
% pbi6=strcat('pb6',num2str(k),'.mat');


save(['result/',cuti], 'cut');
%save(lens,'leng');
%save(accuracy,'acc');
save(['result/',indi], 'Indices');
save(['result/',EPI],'EP');
% save(EPI3,'EP3');
% save(EPI4,'EP4');
% save(EPI5,'EP5');
% save(EPI6,'EP6');
save(['result/',pbi],'PB');
% save(pbi3,'PB3');
% save(pbi4,'PB4');
% save(pbi5,'PB5');
% save(pbi6,'PB6'); %共12个
end
total_time2 = clock;
total_time = etime(total_time2, total_time1)



