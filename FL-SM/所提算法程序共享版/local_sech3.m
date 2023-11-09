
function  NPEFF=local_sech3(AC,M,k,X)

NPEFF=[];
% ACP=sel_pareto(AC,M,k);
ACP=AC;
s1=size(AC,1);

s2=size(ACP,1);
aa=ceil(s2*rand);
refs=ACP(aa,1:k);
addr1=find(refs==1);
s3=size(addr1,2);
s4=ceil(rand*s3);   %随机选的值为1的位

addr0=find(refs==0);
s6=size(addr0,2);
s5=ceil(rand*s6);  %随机选的值为0的位


NEW1=refs;
NEW2=refs;
NEW1(addr1(s4))=0;NEW1(addr0(s5))=1;
NEW2(addr1(s4))=0;

N1EFF=evaluation(NEW1,k,X);
N2EFF=evaluation(NEW2,k,X);
if abs(N1EFF(k+2)-N2EFF(k+2))<abs(N2EFF(k+2)-ACP(aa,k+2))
    flg1=addr1(s4);flg2=addr0(s5);
else
    flg1=addr0(s5);flg2=addr1(s4);
end
NPEFF=[N2EFF;N1EFF];
for jj=1:s1
    cand=AC(jj,1:k);
    if cand(flg1)==1 & cand(flg2)==1
        cand(flg2)=0;
    elseif cand(flg1)==0 & cand(flg2)==0
        cand(flg1)=1;
    elseif  cand(flg1)==1 & cand(flg2)==0
        cand(flg1)=0;
    else
        cand(flg2)=0; cand(flg1)=1; 
    end
    
    CEFF=evaluation(cand,k,X);
    if (CEFF(k+1)>=AC(jj,k+1) & CEFF(k+2)>AC(jj,k+2)) | (CEFF(k+1)>AC(jj,k+1) & CEFF(k+2)>=AC(jj,k+2))
        NPEFF=[NPEFF;AC(jj,:)];
    elseif (CEFF(k+1)<AC(jj,k+1) & CEFF(k+2)<=AC(jj,k+2)) | (CEFF(k+1)<=AC(jj,k+1) & CEFF(k+2)<AC(jj,k+2))
        NPEFF=[NPEFF;CEFF];
    else
        NPEFF=[NPEFF;CEFF;AC(jj,:)];
    end
% NPEFF=[NPEFF;CEFF];
end
















