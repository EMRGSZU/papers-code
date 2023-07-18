function Bbegin(data)
% 功能：开始进行特征选择，主函数！
% 输入：data——数据集，行为实例，列为属性。
% 输出：classnum——类标签的数量
%       classlabel——类标签

%数据预处理将类标签变为最后一列
D=size(data,2)-1;
data=[data(:,2:end) data(:,1)];
data(:,1:end-1)=mapminmax(data(:,1:end-1)',0,1)';
datalabel=data(:,end);
%x=zeros(10,D);
%将数据集进行10倍cv
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



