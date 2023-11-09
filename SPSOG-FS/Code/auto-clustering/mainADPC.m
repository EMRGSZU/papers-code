% clear all; close all; clc;
X=fea;
Y=gnd;
X = (X-min(X))./(max(X)-min(X));
%% Settings of System Parameters for DensityClust
dist = pdist2(X, X);

% for j = 1:8
    j = 2;
    kernel = 'Gauss';
    [~, rho] = paraSet(dist, j/100, kernel);
    [numClust,clustInd,centInd] = ADPC(rho,dist);
    [bacc,~,~] = ClusteringMeasure(clustInd,Y);

% end