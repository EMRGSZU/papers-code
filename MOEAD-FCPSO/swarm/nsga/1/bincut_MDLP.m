function bincut_MDLP(data,cut,l_index,r_index)
% 功能：递归的进行切点选择
% 输出：data--数据集
%       cut-切点坐标
% 

global disc;
bestcutindex=getbestcut(data,cut,l_index,r_index);
    
flag=isMDLPAccepted(data,cut,l_index,r_index,bestcutindex);
if flag==1
    disc=[disc cut(bestcutindex)];
    bincut_MDLP(data,cut,l_index,bestcutindex)
    bincut_MDLP(data,cut,bestcutindex,r_index)	
end