function U=create(ppop,k,F,CR)
%%pop里存放的是第一层上的点
M=2;
% CR=0.1;%%CR取值在[0,1];
U=[];
s1=size(ppop,1);
ALT=0.01;

for i=1:s1
    p=randperm(s1);%%p中存储的是随机打乱顺序的1,2,3,4,5...序列
    h=ceil(rand*k);
    addr=find(p==i);
    if (addr+3)<=s1 %%保证不会超出max_sizeP
        SET1=ppop([p(addr+1:addr+3)],:);
    else
        SET1=ppop([p(addr-3:addr-1)],:);
    end
    
    new_AC=sel_pareto(SET1,M,k);
    if size(new_AC,1)==1
        BAS_V=new_AC;
    else
        BAS_V=sim_deep(new_AC,ppop,k);
    end
    
    for ii=1:3
        if SET1(ii,k+1)==BAS_V(1,k+1)& SET1(ii,k+2)==BAS_V(1,k+2)
            SET1(ii,:)=[];
            break;
        end
    end  
    bb1=0;
    for j=1:M
        if BAS_V(1,k+j)<ppop(i,k+j)
            bb1=bb1+1;
        end
    end
    V= BAS_V(1,1:k);
    if bb1==M
        P=ones(1,k)*ALT;
    else
        P=abs(F*rand*(SET1(1,1:k)-SET1(2,1:k)))+ALT;
    end
    for j=1:k
        if P(1,j)>rand
            V(1,j)=1-V(1,j);  
        end
    end
    U(i,:)=crossProb(ppop(i,:),V,CR,k);
end


function U=crossProb(par,V,CR,k)%%交叉
h=ceil(rand*k);
U=[];

for j=1:k
    uu=rand;
    if ((uu<CR)|(j==h))
        U(1,j)=V(1,j);
    else
        U(1,j)=par(1,j);
    end
end
if sum(U(1:k))==0
    aa=randperm(k);
    for i=1:k
        if rand<0.5
            U(1,aa(i))=1;
            break;
        end
    end
end


        