function bestACC = UDFS20(fea,gnd,FeaNumCandi)
% [fea,gnd] = deal(fea_trans',gnd_trans');
bestACC = zeros(length(FeaNumCandi),1);
nKmeans = 20;
nClass = length(unique(gnd));

[~,~,idx] = UDFS(fea,nClass);
UDFS_idMat = idx;
save("F:\Users\cnnyl\Desktop\2dealt\UDFS\AMDV16_ID_Isolet.mat",'UDFS_idMat')
for feaIdx = 1:length(FeaNumCandi)
    feaNum = FeaNumCandi(feaIdx);
    newfea = fea(:,idx(1:feaNum));
    arrACC = zeros(nKmeans,1);
    for i = 1:nKmeans
        label = litekmeans(newfea,nClass,'Replicates',20);
        [arrACC(i),~,~] = ClusteringMeasure(gnd,label);
    end
    maxACC = max(arrACC);
%     if maxACC > bestACC(feaIdx)
    bestACC(feaIdx) = maxACC;
end
end