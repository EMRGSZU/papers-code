function cut=initcut(data)
% 功能：求取初始的候选离散分割断点
% 输入：data—— n*2 矩阵，第一列待离散的条件属性,第二列为类标签
% 输出：cut——行向量，初始的候选离散分割断点（离散间隔包括左断点,不包括右断点）
initcutp=[];
tmp=sortrows(data,1); %将特征值进行排序
a=tmp(1:end-1,end); %复制两列错开
b=tmp(2:end,end);
cindex=find(a~=b);  %找到相邻之间值不同的点，取他们的平均值作为切点
length=size(cindex,1);
for i=1:length
    colu=cindex(i);
    initcutp=[initcutp (tmp(colu,1)+tmp(colu+1,1))/2];
end
cut=initcutp;


