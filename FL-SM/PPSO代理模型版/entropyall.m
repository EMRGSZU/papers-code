function entSA=entropyall(data)
% 功能：求集合S的entropy
% 输入：data—— 数据集，最后一列为数据集标签，为方便测试，暂不调用函数计算数据标签个数。
%      默认标签个数为4 为0~3
% 输出：entSA—— 整个数据集的Entropy
entS=0;
m=size(data,1);
for i=0:3
    p=size(find(data(:,end)==i),1)/m;
    entS=entS-p*log2(p);
end
entSA=entS;

 

