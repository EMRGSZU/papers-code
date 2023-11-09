function [ErClassification]=GH_accuracy...
    (SClass,TestLables)

NTest=size(TestLables,2);
ErClassification=zeros(1,1);
%ErCls = zeros(1,Ncla);
%NclasSum = zeros(1,Ncla);
%DClass = zeros(1,Ncla);
%for j = 1:NTest
%    flag = 0;
%    if TestLables(1,j) == SClass(1,j)
%         flag = 1;      %用于统计分类正确的样本
%    end
%    for i = 1:Ncla
%        if TestLables(1,j)==(i-1)    
%           NclasSum(1,i)= NclasSum(1,i) + 1 ;%计算训练集中出现的标签总数
%        end
%        if flag == 1 && TestLables(1,j) ==(i-1) 
%               DClass(1,i) = DClass(1,i) + 1;  %统计分类正确的样本
%               flag = 0;
%        end
%    end
%end
%ErCls = ones(1,Ncla) - DClass./NclasSum;
%ErCls = ErCls.*100;
precision = sum(SClass == TestLables);

ErClassification= precision / NTest;
ErClassification= ErClassification * 100;





