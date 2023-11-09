X=fea; Y=gnd;
X = (X-min(X))./(max(X)-min(X));
j = 0.33;
[numClust,clustInd,centInd] = EADPC(X,j);
    