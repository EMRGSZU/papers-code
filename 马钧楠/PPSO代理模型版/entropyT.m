function entST=entropyT(data,cut,point)
% 功能：求取分割后的Entropy
% 输入：data—— n*2 矩阵，第一列待离散的条件属性,第二列标签
%       cut——分割点行向量
%       point——分割点坐标
% 输出：entS——左右离散分割断点之间的熵

entS=0;

lrdata=getlrdata(data,cut,l_index,r_index);
n_lrdata=size(lrdata,1);
[num,value]=attrvalue(lrdata(:,2));    
for j=1:num
    p=size(find(lrdata(:,2)==value(j)),1)/n_lrdata;
    entS=entS-p*log2(p);
end    

