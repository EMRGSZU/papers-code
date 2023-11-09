% clear
% clc
tic

dim = size(fea,2);
if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end
nKmeans = 20;
bestACC = zeros(length(FeaNumCandi),1);
nClass = length(unique(gnd));

alpha = 0.1;
lambda = 0.1;
beta = 1;
NITER = 10;

% URAFS_goodpara =URAFS20test(fea,gnd,FeaNumCandi);
[~,~,index] = URAFS(fea,nClass,alpha,beta,lambda,NITER);

for feaIdx = 1:length(FeaNumCandi)
    feaNum = FeaNumCandi(feaIdx);
    newfea = fea(:,index(1:feaNum));
    arrACC = zeros(nKmeans,1);
    parfor i = 1:nKmeans
        label = litekmeans(newfea,nClass,'Replicates',20);
        [arrACC(i),~,~] = ClusteringMeasure(gnd,label);
    end
    maxACC = max(arrACC);
%     if maxACC > bestACC(feaIdx)
    bestACC(feaIdx) = maxACC;
   
end
% URAFS_result = bestACC;
% save("F:\Users\cnnyl\Desktop\2dealt\URAFS\USPS.mat",'URAFS_result')
toc


%     path = strcat(".\mytest\newpara\test\RSFS20\",'RSFS20',num2str(tryb),'.mat');
%     save(path,'accscore')
% end