function [Y,Min,Max]=MINMAX(X)
    [M,N]=size(X);
    Min=min(X);
    Max=max(X);
    Y=X-repmat(Min,[M,1]);
    Y=Y./repmat(Max-Min,[M,1]);
end