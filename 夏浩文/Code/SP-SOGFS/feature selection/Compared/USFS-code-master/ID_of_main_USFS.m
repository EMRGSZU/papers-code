% clc;
% clear;
% load('COIL20.mat')
tic
warning('off');
dim = size(fea,2);
X = fea';label = gnd;
nClass = length(unique(label));
[d, n] = size(X);
rand('twister',5489);

if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end
nKmeans = 20;
bestACC = zeros(length(FeaNumCandi),1);

alpha = 1000;
gamma = 1000;
beta = 1;
m = 45;
%% run SLUFS
[W, Y, P, M] = USFS(X, nClass, m, alpha, beta,gamma);
W1 = [];
for k = 1:d
    W1 = [W1 norm(P(k,:),2)];
end
[~,index] = sort(W1,'descend');

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
USFS_result = bestACC;
save("F:\Users\cnnyl\Desktop\2dealt\USFS\USPS.mat",'USFS_result')
toc
% new_fea = X(index(1:FeaNumCandi),:);
% idx = kmeans(new_fea', class_num);
% result = ClusteringMeasure(label, idx);