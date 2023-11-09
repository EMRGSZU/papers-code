tic
X = fea;
Y = gnd;
% X = (X-min(X))./(max(X)-min(X));

gamma = [0.001,0.01,0.1,1,10,100,1000];
k = 31;
nClass = length(unique(gnd));
isToPara = 1;

%SOGFS&SOGFS-SP
X = X';
num = size(X,2);
dim = size(X,1);

if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end
feaLen = length(FeaNumCandi);


%%%数据矩阵预处理
X0 = X';
mX0 = mean(X0);
X1 = X0 - ones(num,1)*mX0;
scal = 1./sqrt(sum(X1.*X1)+eps);
scalMat = sparse(diag(scal));
X = X1*scalMat;
X = X';%让数据的每个特征归一化——同一特征减去平均值再单位化

distX = L2_distance_1(X,X); %计算原始数据两两数据间的欧式距离的平方
[distX1, idx] = sort(distX,2);
toc
%%
tic
if isToPara == 0
    gamma = 1;
    k =70;
    id = SOGFS_sp2(X,distX1,idx,k,gamma,nClass,num,dim);
    paraAndResult = [1:feaLen, 0, gamma;
                     para(id,FeaNumCandi,fea,gnd,nClass), 0,k];
else
    feaIdx = 1:(feaLen);
    resultMat = zeros(length(gamma)*(k-1),feaLen);
    for iIndex = 1:length(gamma)
        i = gamma(iIndex);
        for j =2:k
            id = SOGFS_sp2(X,distX1,idx,j,i,nClass,num,dim);
            resultMat((iIndex-1)*(k-1)+j-1,:) = para(id,FeaNumCandi,fea,gnd,nClass);
        end
    end
    [bestAcc,accInd] = max(resultMat);    
end
toc

save("F:\Users\cnnyl\Desktop\SP-SOGFS\feature selection\Result\SP\UMist.mat",'bestAcc','accInd','resultMat')

function oneParaResult = para(id,FeaNumCandi,fea,gnd,nClass)
    len = length(FeaNumCandi);
    oneSetAcc = 1:len;
    for feaIdx = 1:len
        feaNum = FeaNumCandi(feaIdx);
        newfea = fea(:,id(1:feaNum));
        arrACC = zeros(20,1);
        parfor i = 1:20
            label = litekmeans(newfea,nClass,'Replicates',20);
            [arrACC(i),~,~] = ClusteringMeasure(gnd,label);
        end
        oneNumACC = max(arrACC);
        oneSetAcc(feaIdx) = oneNumACC;
    end
    oneParaResult =oneSetAcc;
end