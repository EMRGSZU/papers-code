function [Bratio,TT]=GH_snapping...
    (i,gamma,TT,patterns,targets,N,knn)
No_C1=0;
for j=1:size(targets,2)
    if targets(1,j)==targets(1,i)
        No_C1=No_C1+1;
    end
end
No_C2=N-No_C1;
No_unq=1;
DR=0;
FAR=0;
[DR, FAR]=GH_radiousRepX(N,1,patterns,targets,TT,gamma,i,knn);%判定分类聚集情况
Bratio=DR/No_C1-FAR/No_C2;%计算分类聚集指数

