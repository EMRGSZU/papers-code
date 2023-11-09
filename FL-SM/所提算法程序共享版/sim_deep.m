
function BAS_V=sim_deep(SET0,ppop,k)

s1=size(SET0,1);
s2=size(ppop,1);

deep=[];
for i=1:s1
    for j=1:s2
        deep(i,j)=sum(abs(SET0(i,1:k)-ppop(j,1:k)));
    end
end

dd=-1;
for i=1:s1
    [val,ind]=sort(deep(i,:));
    deep2=sum(val(1:2));
    if deep2>dd
        dd=deep2;
        flg=i;
    end
end

BAS_V=SET0(flg,:);