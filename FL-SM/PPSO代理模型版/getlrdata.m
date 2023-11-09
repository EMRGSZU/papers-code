function lrdata=getlrdata(data,cut,l_index,r_index)
% 功能：求取左右离散分割断点之间的数据
% 输入：data—— n*2 矩阵，第一列待离散的条件属性,第二列决策属性
% 输出：lrdata——左右离散分割断点之间的数据
             %6
if r_index>size(cut,2) %切点个数
    tmpindex=1:size(data,1);  %1~样本数量
else
    tmpindex=find(data(:,1)<cut(r_index));   
end
tmpdata=data(tmpindex,:);  %所有比第一个点晓得数
if l_index<1
    tmp=min(data(:,1));  %此特征中最小值
    lrdata=tmpdata(find(data(tmpindex,1)>=tmp(1)),:);
else
    lrdata=tmpdata(find(data(tmpindex,1)>=cut(l_index)),:);
end
