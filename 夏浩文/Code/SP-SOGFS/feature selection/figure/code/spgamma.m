function start = spgamma(fea,gnd,d2,dataset)
nClass = length(unique(gnd));
gammacnadi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
if d2 == 1
    NewFeaNum= 10:10:100;
else
    NewFeaNum = 50:50:300;
end

len= length(NewFeaNum);

%create a folder named by the name of dataset
datadir = strcat("F:\Users\cnnyl\Documents\MATLAB\mytest\SPSOGFS\test\gamma_output\",dataset);
if exist(datadir,'dir') == 0
    mkdir datadir;
end

for gamma = gammacnadi
    result_path = strcat(datadir,'\','gamma_',num2str(gamma),'_result.mat');
    idx= SOGFS_sp(fea,gamma,nClass,55);%1.1 63
    fea1 = fea';
    mtrResult = [];
    
    for i=1:len
        feaNum = NewFeaNum(i);
        Newfea=fea1(:,idx(1:feaNum));
        
        
        for iter=1:20
            label=litekmeans(Newfea,nClass,'Replicates',20);
            [Acc(iter),~]=ClusteringMeasure(gnd,label);
        end
        maxacc = max(Acc);
        mtrResult = [mtrResult,[feaNum,maxacc]'];
    end
    save(result_path,'mtrResult');
end
start = 1;
end