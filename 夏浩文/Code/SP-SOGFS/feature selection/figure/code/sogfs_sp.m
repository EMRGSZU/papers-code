function spsogfsm = sogfs_sp(fea,gnd,d2)
nClass = length(unique(gnd));

if d2 == 1
    NewFeaNum= 10:10:100;
else
    NewFeaNum = 50:50:300;
end

len= length(NewFeaNum);

idx= SOGFS_sp(fea,1,nClass,66);%1.1 63
fea1 = fea';
spsogfsm(1,len) = 0;
for i=1:len
    Newfea=fea1(:,idx(1:NewFeaNum(i)));
    for iter=1:20
        label=litekmeans(Newfea,nClass,'Replicates',20);
        [Acc(iter),~]=ClusteringMeasure(gnd,label);
    end
    spsogfsm(1,i)=max(Acc);
end
