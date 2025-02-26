function [fstar,fstarLin,ErCls,rClassification,SClass] = LFS(Train, TrainLables, Test, TestLables, Para,balanced)
%% parameters
gamma=Para.gamma;
tau=Para.tau;
alpha=Para.alpha;
sigma=Para.sigma;
knn=1;%最近的knn个点
classlabel=unique(TrainLables);
N=size(Train,2);
M=size(Train,1);
fstar=zeros(M,N);
fstarLin=zeros(M,N);

clsNum=zeros(1,numel(classlabel));
for i=1:numel(classlabel)
    clsNum(i)=sum(TrainLables==classlabel(i));
end

[~,FWeights]=relieff(Train',TrainLables',5);
minW=min(FWeights);
maxW=max(FWeights);
FWeights=(FWeights-minW)/(maxW-minW);
for j=1:tau
    [b, a]=GH_FES...
        (M,N,Train,TrainLables,fstar,sigma);%计算距离权重a，b
    
      [TBTemp,TRTemp]=GH_evaluation...%粒子群算法求解 我们的方法
          (alpha,N,b,a,Train,TrainLables,gamma,knn,FWeights);
    
    fstar=TBTemp;
    fstarLin=TRTemp;
    
end

[SClass]=GH_Classification...
    (Train,TrainLables,N,Test,fstar,gamma,knn,classlabel);%进行分类测试

[ErCls,rClassification]= ...
    GH_accuracy(SClass,TestLables,classlabel,balanced);%计算分类错误率
end