function [leng,err]=rbegin(data,Indices,EP)
% 功能：开始进行特征选择，主函数！
% 输入：data——数据集，行为实例，列为属性。
% 输出：classnum——类标签的数量
%       classlabel——类标签
%

%数据预处理将类标签变为最后一列
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
% x=[];
%将数据集进行10倍cv
% Indices =crossvalind('Kfold',datalabel, 10);
for i=1:10
    disp(i)
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
cut=disc_MDLP(traind);
    for j=1:size(EP{1,i},1)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %返回特征选择结果
    x=EP{1,i}(j).Position();
    [leng(i,j),err(i,j)] = testerror(x,traind,testd,cut); %查看选择特征数量和分类正确率
    end
end
save matlab1 leng ;
save matlab2 err ;

