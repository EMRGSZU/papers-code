function entS=entropy_interval(data,cut,l_index,r_index)
% 功能：求取左右离散分割断点之间的熵
% 输入：data—— n*2 矩阵，第一列待离散的条件属性,第二列决策属性
%       cut——行向量，初始的候选离散分割断点（离散间隔包括左断点,不包括右断点）
%       l_index——左离散分割断点在cut中的索引，0 代表包含最左断点左侧的数据
%       r_index——右离散分割断点在cut中的索引，size(cut,2)+1 代表包含最右断点右侧的数据
% 输出：entS——左右离散分割断点之间的熵

entS=0;

lrdata=getlrdata(data,cut,l_index,r_index);
n_lrdata=size(lrdata,1);
[num,value]=attrvalue(lrdata(:,2));    
for j=1:num
    p=size(find(lrdata(:,2)==value(j)),1)/n_lrdata;
    entS=entS-p*log2(p);
end    

