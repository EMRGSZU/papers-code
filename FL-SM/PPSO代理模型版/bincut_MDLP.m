function bincut_MDLP(data,cut,l_index,r_index)


global disc;
bestcutindex=getbestcut(data,cut,l_index,r_index);
    
flag=isMDLPAccepted(data,cut,l_index,r_index,bestcutindex);
if flag==1
    disc=[disc cut(bestcutindex)];
    bincut_MDLP(data,cut,l_index,bestcutindex)
    bincut_MDLP(data,cut,bestcutindex,r_index)	
end