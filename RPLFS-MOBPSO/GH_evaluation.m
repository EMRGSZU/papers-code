function [TBTemp,TRTemp]=GH_evaluation...
    (alpha,N,b,a,patterns,targets,gamma,knn,FWeights)
M=size(b,2);

NBeta=100;%解集最大容量
TRTemp=zeros(M,N);
TBTemp=zeros(M,N);

parfor i=1:N
    warning off
    [TTPrato]=MOBPSO_NSGA3(a(i,:),b(i,:),M,alpha,FWeights,@(position)fitness(i, patterns', targets', position));

    min_tmp = 1;
    I1 = 1;
    for j=1:numel(TTPrato)
        if TTPrato(j).Cost(3)<min_tmp
            min_tmp = TTPrato(j).Cost(3);
            I1 = j;
        end
    end
    TBTemp(:,i)=(TTPrato(I1).Position>0.5)';
    TRTemp(:,i)=(TTPrato(I1).Position)';

end
end









