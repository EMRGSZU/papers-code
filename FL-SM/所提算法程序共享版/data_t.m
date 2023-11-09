
function X=data_t(fly)
fly=1;
    load('data/9_Tumors.mat')
    for i=1:size(X,2)
    max1=max(X(:,i));
    min1=min(X(:,i));
    X(:,i)=(X(:,i)-min1)/(max1-min1)+0.0001;
    end 
    X=X;
    Y=Y;
    X=[Y,X];