function [S_Class]=GH_Classification...
    (patterns,targets,N,test,fstar,gamma,knn,Ncla)
H=size(test,2);
%S_Class1_sphereKNN=zeros(1,H);
%S_Class2_sphereKNN=zeros(1,H);
S_Class = zeros(1,H);
for t=1:H
%    [S_Class1_sphereKNN(1,t), S_Class2_sphereKNN(1,t),~]=...
%        GH_ClassSimM(test(:,t),N,patterns,targets,fstar,gamma,knn);
    [S_Class(1,t),~]=...
            Class(test(:,t),N,patterns,targets,fstar,gamma,knn,Ncla);
end
end


