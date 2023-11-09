function bestACC = RSFS20(fea,gnd,FeaNumCandi,para)
%run RSFS
% [nSmp,nFea] = size(fea);% fea : num*dim
%======================setup=======================
% [fea,gnd] = deal(fea_trans',gnd_trans');
bestACC = zeros(length(FeaNumCandi),1);
nKmeans = 20;
nClass = length(unique(gnd));

% alfaCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
% betaCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
% nuCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
alfaCandi = para(2);
betaCandi = para(3);
nuCandi = para(4);

maxIter = 20;
S = constructW(fea); 
nRowS = size(S,1);

for i = 1:nRowS
    sum_row = sum(S(i,:));
    S(i,:) = S(i,:)/sum_row;
end

diag_ele_arr = sum(S+S',2);
A = diag(diag_ele_arr);
L = A-S-S';

eY = eigY(L,nClass);
rand('twister',5489);
label = litekmeans(eY,nClass,'Replicates',20);
Y_init = zeros(size(fea,1),nClass);
for i = 1:size(fea,1)
    Y_init(i,label(i)) = 1;
end

%Clustering using selected features
for alpha = alfaCandi
    for beta = betaCandi
        for nu = nuCandi
            Y = Y_init;
%             disp(['alpha=',num2str(alpha),',','beta=',num2str(beta),',','nu=',num2str(nu)]);
%             result_path = strcat(dataset,'\','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'_result.mat');
            mtrResult = [];
            Z = zeros(size(fea,1),length(unique(gnd)));
            W = RSFS(fea,L,Z,Y,alpha,beta,nu,maxIter);

            [dumb idx] = sort(sum(W.*W,2),'descend');
            
            for feaIdx = 1:length(FeaNumCandi)
                feaNum = FeaNumCandi(feaIdx);
                newfea = fea(:,idx(1:feaNum));
                rand('twister',5489);
                arrACC = zeros(nKmeans,1);
                for i = 1:nKmeans
                    label = litekmeans(newfea,nClass,'Replicates',20);
                    arrACC(i) = ACC_Lei(gnd,label);
                end
                maxACC = max(arrACC);
%                 if maxACC > bestACC(feaIdx)
                bestACC(feaIdx) = maxACC;
%                 end
%                 mtrResult = [mtrResult,maxACC];
            end
%             save(result_path,'mtrResult');
        end
    end
end




end