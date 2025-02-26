function [S_Class,radious]=GH_ClassSimM(test,N,...
    patterns,targets,fstar,gamma,knn,classlabel)

M=size(patterns,1);
%%
classnum=numel(classlabel);

NoNNCknn=zeros(classnum,N);
nearestPoints=zeros(classnum,N);
VotingRights=zeros(1,N);
flag=0;%标志是否进行过投票
radious=zeros(1,N);
for i=1:N
    %%
    NTrClsl=0;
    targetsC1=zeros(1,N);%计算同类位置
    for j=1:size(targets,2)%计算同类数量
        if targets(1,j)==targets(1,i)
            NTrClsl=NTrClsl+1;
            targetsC1(1,j)=1;
        end
    end
    NTrCls2=N-NTrClsl;%不同类数量
    XpatternsPr=patterns.*repmat(fstar(:,i),1,N);
    testPr=test.*fstar(:,i);
    Dist=abs(sqrt(sum((XpatternsPr-repmat(testPr,1,N)).^2,1)));
    [min , minI]=sort(Dist,2);
    
    min_Uniq=unique(min);
    m=0;
    No_nereser=0;
    while No_nereser<knn
        m=m+1;
        a1=min_Uniq(m);
        NN=Dist<=a1;
        No_nereser=sum(NN);
    end
    NNIndex=find(NN==1);
    temp_NoNNCknn=zeros(classnum,1);
    for j=1:numel(NNIndex)%统计test点附近k个点的类
        temp_knnClsIndex=targets(1,NNIndex(1,j))==classlabel;
        temp_NoNNCknn(temp_knnClsIndex,1)= temp_NoNNCknn(temp_knnClsIndex,1)+1;
        nearestPoints(temp_knnClsIndex,i)= 1;
    end
    [~,temp_index]=max(temp_NoNNCknn);
    NoNNCknn(temp_index,i)=NoNNCknn(temp_index,i)+1;%更新位图，统计弱分类信息
    %%
    A=find(fstar(:,i)==0);
    if length(A)<M
        patterns_P=patterns;
        patterns_P(A,:)=[];
        test_P=test;
        test_P(A,:)=[];
        Dist_test=abs(sqrt(sum((patterns_P(:,i)-test_P).^2,1)));
        Dist_pat=abs(sqrt(sum((patterns_P-repmat(patterns_P(:,i),1,N)).^2,1)));
        [EE_Rep , ~]=sort(Dist_pat);
        remove=0;
        UNQ=unique(EE_Rep);
        k=0;
        if remove~=1
            Next=1;
            while Next==1%计算最大代表区域
                k=k+1;
                r=UNQ(k);
                F2=(Dist_pat<=r) ;
                NoCls1clst=sum(and(targetsC1,F2))-1;
                NoCls2clst=sum(and(F2,not(targetsC1)));
                if    gamma*(NoCls1clst/(NTrClsl-1))<(NoCls2clst/NTrCls2) || k==numel(UNQ)%判断满足代表区域条件
                    Next=0;
                    if (k-1)==0 || (k==numel(UNQ) && gamma*(NoCls1clst/(NTrClsl-1))>=(NoCls2clst/NTrCls2))
                        r=UNQ(k);
                    else
                        r=0.5*(UNQ(k-1)+UNQ(k));
                    end
                    if r==0
                        r=1e-6;
                    end
                end
            end
            maxF=temp_NoNNCknn==max(temp_NoNNCknn);
            if sum(maxF)==1 && Dist_test<=r && targets(1,i)==classlabel(1,maxF)%在代表区域内且临近点多数与代表点同类则投票
                VotingRights(1,i)=1;
                flag=1;
            end
        end
    end
end

classcount=zeros(classnum,1);
for j=1:classnum
    classcount(j,1)=sum((targets==classlabel(1,j)));
end

if flag==1 %有投票权的情况
    [~,S_Class]=max(sum(NoNNCknn(:,VotingRights==1),2)./classcount,[],1);
else %没有投票权的情况
    [~,S_Class]=max(sum(nearestPoints,2)./classcount,[],1);
end

if numel(S_Class)~=1%如果出现多个类投票数均最多，则从中随机选择一个结果
S_Class=S_Class(randi(1,1,numel(S_Class)));%[warning!!!如果测试点不在任何球中，就会随机从全体类中选择类标签]
end
S_Class=classlabel(1,S_Class);



