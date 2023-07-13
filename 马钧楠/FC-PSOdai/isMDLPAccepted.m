function flag=isMDLPAccepted(data,cut,l_index,r_index,cutindex)
% 功能：求是否符合MDLP准则的分割点
% 输入：data—— n*2 矩阵，第一列待离散特征值,第二列为类标签
%       cut--切点
% 输出：flag为0代表不接受1代表接受
if cutindex==0
    flag=0;
    return;
end

N=r_index-l_index;
lN=cutindex-l_index;
rN=r_index-cutindex;

lrdata=getlrdata(data,cut,l_index,r_index);
N=size(lrdata,1);
[k,~]=attrvalue(lrdata(:,2));    
ldata=getlrdata(data,cut,l_index,cutindex);
[k1,~]=attrvalue(ldata(:,2));    
lN=size(ldata,1);
rdata=getlrdata(data,cut,cutindex,r_index);
[k2,~]=attrvalue(rdata(:,2));    
rN=size(rdata,1);

entS=entropy_interval(data,cut,l_index,r_index);
lentS=entropy_interval(data,cut,l_index,cutindex);
rentS=entropy_interval(data,cut,cutindex,r_index);

MDLP=entS*N-lentS*lN-rentS*rN+entS*k-lentS*k1-rentS*k2-(log2(N-1)+log2(power(3,k)-2));
if MDLP>0
    flag=1;
else
    flag=0;
end
