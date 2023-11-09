
    
function NPEFF=local_sech2(AC,M,k,X)


s1=size(AC,1);
lsize=s1;
crowd_value=calcul_crowd(AC,M,k);
[val,ind]=sort(crowd_value);
BVT=AC(ind(s1),:);

newpop=[];
for i=1:lsize
    a1=ceil(rand*s1);
    did1=abs(BVT(1:k)-AC(a1,1:k));
    ff1=(1-0.5*(abs(BVT(k+2)+AC(a1,k+2))))/sum(did1);
    
    a2=ceil(rand*s1);
    did2=abs(BVT(1:k)-AC(a2,1:k));
    ff2=(1-0.5*(abs(BVT(k+2)+AC(a2,k+2))))/sum(did2);
    
    if ff1>ff2
        cand=AC(a1,:);
        flg=did1;
    else
        cand=AC(a2,:);
        flg=did2;
    end
    
    addr=find(flg==1);
    s3=size(addr,1);
    if s3>1
        for hh=1:s3
            cand(addr(hh))=round(rand);
        end
    else
        addr1=find(flg==0);
        s6=size(addr1,1);
        s7=ceil(rand*s6);
        cand(addr1(s7))=round(rand);
    end
    newpop=[newpop;cand];
end
NPEFF=evaluation(newpop,k,X);


