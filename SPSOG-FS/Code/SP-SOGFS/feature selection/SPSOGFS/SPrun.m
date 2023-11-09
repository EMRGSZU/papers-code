tic
X = fea;
Y = gnd;
gamma = 1;
k = 8;

X = X';
dim = size(fea,2);
if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end

nKmeans = 20;
bestACC = zeros(length(FeaNumCandi),1);
nClass = length(unique(gnd));

idx = SOGFS_sp_origin(X,gamma,nClass,k);
for feaIdx = 1:length(FeaNumCandi)
    feaNum = FeaNumCandi(feaIdx);
    newfea = fea(:,idx(1:feaNum));
    arrACC = zeros(nKmeans,1);
    parfor i = 1:nKmeans
        label = litekmeans(newfea,nClass,'Replicates',20);
        [arrACC(i),~,~] = ClusteringMeasure(gnd,label);
    end
    maxACC = max(arrACC);
%     if maxACC > bestACC(feaIdx)
    bestACC(feaIdx) = maxACC;
end
SP_result = bestACC;
    save("F:\Users\cnnyl\Desktop\2dealt\SP\ecoli.mat",'SP_result')
    toc