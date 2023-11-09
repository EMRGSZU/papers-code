function [max_index, radious]=Class(test,N,...
    patterns,targets,fstar,gamma,knn,Ncla) %测量每个测试点的相似性
%%
NC = zeros(1,Ncla + 1);
S= -1*ones(1,N);
NC(1,1) = inf;
NoNNCknn=zeros(1,N);
radious=zeros(1,N);
for t = 2:Ncla + 1
    NC(1,t) = sum(targets == (t-2));     %统计每个样本出现的总次数，目的是为了等下做正则化
end
parfor i=1:N
    %%
    %为了方便等下计算KNN的
    cls1 = targets(1,i);
    XpatternsPr=patterns.*repmat(fstar(:,i).position.',1,N); %所有样本点在该选定特征集上的值
    testPr=test.* fstar(:,i).position.';
    Dist=abs(sqrt(sum((XpatternsPr-repmat(testPr,1,N)).^2,1)));
    [min , ~]=sort(Dist,2);
    min_Uniq=unique(min);
    m=0;
    No_nereser=0;
    while No_nereser<knn
        m=m+1;
        a1=min_Uniq(m);
        NN=Dist<=a1;
        No_nereser=sum(NN);
    end
    nearest_NN = targets(1,NN);
    nearest_diff_num = sum(nearest_NN ~= cls1);
    b = 0:Ncla-1; %这是为下标为0准备的，如果没有下标为0,则会比较有一些问题
    [~ ,max_index] = max(histc(nearest_NN,b));
    NoNNCknn(1,i) = max_index - 1;
    
    NTrClsl=sum(targets == cls1);
    NTrCls2=N-NTrClsl; 
    %%
    A=find(fstar(:,i).position.'==0); %除去没被选中的那部分特征
    if NTrClsl > 1
        patterns_P=patterns;
        patterns_P(A,:)=[];
        test_P=test;
        test_P(A,:)=[];
        Dist_test=abs(sqrt(sum((patterns_P(:,i)-test_P).^2,1))); %测试test_point和所有点在fstar坐标空间之间的距离
        Dist_pat=abs(sqrt(sum((patterns_P-repmat(patterns_P(:,i),1,N)).^2,1))); %训练样本之间在坐标空间fstar之间的距离
        [EE_Rep , ~]=sort(Dist_pat);
        remove=0;
        UNQ=unique(EE_Rep);
        k=0;
        if remove~=1
            Next=1;
            while Next==1
                k=k+1;
                r=UNQ(k);
                F2=(Dist_pat<=r); %确定超球体半径
                NoCls1clst=sum(targets(1,F2)==cls1);  %减一是为了减去中心位置
                NoCls2clst=sum(F2) - NoCls1clst; %类别为0
                if    gamma*(NoCls1clst-1)/(NTrClsl-1)<(NoCls2clst/NTrCls2) %球体内不同类别和相同类别的比例(最接近边缘有最大的比例可以计算)
                    Next=0;
                    if (k-1)==0
                        r=UNQ(k);
                    else
                        r=0.5*(UNQ(k-1)+UNQ(k));
                    end
                    if r==0
                        r=1e-6;
                    end
                    r=1*r;
                    F2=(Dist_pat<=r);
                    NoCls1clst=sum(targets(1,F2)==cls1); 
                    NoCls2clst=sum(F2) - NoCls1clst;
                end
            end
            if Dist_test<=r   %确定测试样本点是否落入到以这个半径的超球体里面/不落入这个球体就默认用knn的计算方式
                patterns_P=patterns.*repmat(fstar(:,i).position.',1,N);
                test_P=test.* fstar(:,i).position.';
                Dist=abs(sqrt(sum((patterns_P-repmat(test_P,1,N)).^2,1)));
                [min , ~]=sort(Dist,2);
                min_Uniq=unique(min);
                m=0;
                No_nereser=0;
                while No_nereser<knn
                    m=m+1;
                    a1=min_Uniq(m);
                    NN=Dist<=a1;
                    No_nereser=sum(NN);
                end
                NoNNC1 = sum(targets(1,NN)==cls1);
                %NoNNC2 = sum(NN) - NoNNC1;%如果离样本最近点的标签是NoNNC1,则相似度为1
                if NoNNC1 > nearest_diff_num  %最近邻必须比不同类别样本多。
                    S(1,i)=cls1;
                end
            end
        end
    end   
    radious(1,i)=r;
 %因为这里有类别0
end
if sum(S) == -1 * N
    b = -1:Ncla-1;  %减一是因为有0的存在，数组长度最长就是16/等更换数组后再调整吧
    S_Class = histc(NoNNCknn,b);
    S_Class = S_Class./NC; 
    [~,max_index] = max(S_Class);    
else
    b = -1:Ncla-1;
    S_Class = histc(S,b);
    S_Class = S_Class./NC;
    [~,max_index] = max(S_Class);
end
max_index = max_index - 2;





