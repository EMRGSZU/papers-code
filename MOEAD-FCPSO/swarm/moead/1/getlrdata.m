function lrdata=getlrdata(data,cut,l_index,r_index)
% 功能：求取左右离散分割断点之间的数据
% 输入：data—— n*2 矩阵，第一列待离散特征值,第二列为类标签
% 输出：lrdata——左右离散分割断点之间的数据
          
if r_index>size(cut,2) 
    tmpindex=1:size(data,1);  
else
    tmpindex=find(data(:,1)<cut(r_index));   
end
tmpdata=data(tmpindex,:);  
if l_index<1
    tmp=min(data(:,1));  
    lrdata=tmpdata(find(data(tmpindex,1)>=tmp(1)),:);
else
    lrdata=tmpdata(find(data(tmpindex,1)>=cut(l_index)),:);
end
