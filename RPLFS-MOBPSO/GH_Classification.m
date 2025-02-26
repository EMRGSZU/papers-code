function [S_Class_sphereKNN]=GH_Classification...
    (patterns,targets,N,test,fstar,gamma,knn,classlabel)
H=size(test,2);
S_Class_sphereKNN=zeros(1,H);
parfor t=1:H%逐个点进行分类，并行处理
    [S_Class_sphereKNN(1,t),~]=...
        GH_ClassSimM(test(:,t),N,patterns,targets,fstar,gamma,knn,classlabel);%计算分类结果
end
end


