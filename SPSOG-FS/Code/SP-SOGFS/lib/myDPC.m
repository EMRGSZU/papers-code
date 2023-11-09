function [delta,indNearNeigh,ordRho] = myDPC(dist,rho)
%MYDPC Clustering by fast search and find of density peaks
%参照其他人的实现，选取自己需要的功能
%Copyright (c) 2015, QiQi Duan / Matlab DensityClust_1.2
%   INPUT:
%       dist: [NE, NE] distance matrix
%       dc: cut-off distance
%       rho: local density [row vector]

%   OUTPUT:
%       numClust: number of clusters
%       clustInd: cluster index that each point belongs to, NOTE that -1 represents no clustering assignment (haloInd points)
%       centInd:  centroid index vector

    [NE, ~] = size(dist);
    delta = zeros(1, NE); % minimum distance between each point and any other point with higher density
    indNearNeigh = Inf * ones(1, NE); % index of nearest neighbor with higher density
    
    [~, ordRho] = sort(rho, 'descend');
 
    for i = 2 : NE
        delta(ordRho(i)) = max(dist(ordRho(i), :));
        for j = 1 : (i-1)
            if dist(ordRho(i), ordRho(j)) < delta(ordRho(i))
                delta(ordRho(i)) = dist(ordRho(i), ordRho(j));
                indNearNeigh(ordRho(i)) = ordRho(j);
            end
        end
    end
    delta(ordRho(1)) = max(delta);
    indNearNeigh(ordRho(1)) = 0; % no point with higher density
    
    isManualSelect = 1; % 1 denote that all the cluster centroids are selected manually, otherwise 0
%     [numClust, centInd] = decisionGraph(rho, delta, isManualSelect);
    
    % after the cluster centers have been found,
    % each remaining point is assigned to the same cluster as its nearest neighbor of higher density
    clustInd = zeros(1, NE);
%     for i = 1 : NE
%         if centInd(ordRho(i)) == 0 % not centroid
%             clustInd(ordRho(i)) = clustInd(indNearNeigh(ordRho(i)));
%         else
%             clustInd(ordRho(i)) = centInd(ordRho(i));
%         end
%     end
%     
% end


