function bestcutindex=getbestcut(data,cut,l_index,r_index)

bestcutindex=0;

lrdata=getlrdata(data,cut,l_index,r_index);
n_lrdata=size(lrdata,1); %第一次分割后的数据长度
tmpig=Inf;
for i=(l_index+1):(r_index-1)
    ig=0;
    ldata_i=getlrdata(data,cut,l_index,i);
    n_ldata_i=size(ldata_i,1);      %列长度
    entS=entropy_interval(data,cut,l_index,i);
    
    
    ig=ig+(n_ldata_i/n_lrdata)*entS;    
    
    rdata_i=getlrdata(lrdata,cut,i,r_index);
    n_rdata_i=size(rdata_i,1);
    entS=entropy_interval(data,cut,i,r_index);
    ig=ig+(n_rdata_i/n_lrdata)*entS; 
    
    if ig<tmpig
        tmpig=ig;
        bestcutindex=i;
    end
end
