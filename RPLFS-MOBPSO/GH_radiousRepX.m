function [DR, FAR]=GH_radiousRepX(N,No_RepPoints,patterns,targets,T,gamma,y,knn)
NTrCls1=0;
targetsC1=zeros(1,N);%计算同类位置
for j=1:N
    if targets(1,j)==targets(1,y)
        NTrCls1=NTrCls1+1;
        targetsC1(1,j)=1;
    end
end
NTrCls2=N-NTrCls1;%不同类数量
M=size(patterns,1);
NC1=0;
DR=0;
FAR=0;
for i=1:No_RepPoints
    A=find(T(:,i)==0);
    if length(A)<M
        RepPoints_P=patterns;
        RepPoints_P(A,:)=[];
        Dist_Rep=abs(sqrt(sum((RepPoints_P-repmat(RepPoints_P(:,y),1,N)).^2,1)));
        [EE_Rep , ~]=sort(Dist_Rep);
        UNQ=unique(EE_Rep);
        k=0;
        Next=1;
        while Next==1%寻找最大代表区域!!!
            k=k+1;
            r=UNQ(k);
            F2=(Dist_Rep<=r);
            NCls1clst=sum(and(F2,targetsC1))-1;
            NCls2clst=sum(and(F2,not(targetsC1)));
            if  gamma*(NCls1clst/(NTrCls1-1))<(NCls2clst/NTrCls2) || gamma*NCls1clst<NCls2clst ||k==numel(UNQ)%判定是否满足条件
%             if  gamma*NCls1clst<NCls2clst || k==numel(UNQ)%判定是否满足条件
                Next=0;
                if (k-1)==0 || (k==numel(UNQ) && gamma*(NCls1clst/(NTrCls1-1))>=(NCls2clst/NTrCls2))
                    r=UNQ(k);
                else
                    r=0.5*(UNQ(k-1)+UNQ(k));
                end
                if r==0
                    r=1e-6;
                end
                [~, Q]=find(F2==1);
                for u=1:size(Q,2)
                    quasiTest_P=RepPoints_P(:,Q(u));
                    Dist_quasiTest=abs(sqrt(sum((RepPoints_P-repmat(quasiTest_P,1,N)).^2,1)));
                    [min , ~]=sort(Dist_quasiTest,2);
                    min_Uniq=unique(min);
                    m=0;
                    No_nereser=0;
                    while No_nereser<knn+1
                        m=m+1;
                        a1=min_Uniq(m);
                        NN=Dist_quasiTest<=a1;
                        No_nereser=sum(NN);
                    end
                    No_NN_C1=sum(and(NN,targetsC1));%knn中同类数量
                    No_NN_C2=sum(and(NN,not(targetsC1)));%knn中不同类数量
                    
                    if targets(1,Q(u))==targets(1,y)&&(No_NN_C1-1)>No_NN_C2%adaptive1
                        DR=DR+1;%同类含量
                    end
                    if targets(1,Q(u))~=targets(1,y)&&No_NN_C1>(No_NN_C2-1)
                        FAR=FAR+1;%不同类含量
                    end

                end
            end
        end
    end
end