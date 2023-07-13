X = (X-min(X))./(max(X)-min(X));
pro = 1;

dist = pdist2(X, X);
kernel = 'Gauss';
[dc, rho] = paraSet(dist, pro/100, kernel);
    
[Num, ~] = size(dist);
delta = zeros(1, Num); % minimum distance between each point and any other point with higher density
indNearNeigh = Inf * ones(1, Num); % index of nearest neighbor with higher density

[~, ordRho] = sort(rho, 'descend');
 
    for i = 2 : Num
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

    indNN =indNearNeigh';
    
    numClust = 1;
    centInd = ordRho(1);
%    
%     while isempty(rho)
%         for m = 1:length(centInd2)
%             center_sup = find(delta >= dc);
%             center_ind = find(indNearNeigh == cnetInd(m));
%             
%         
    cluster_t = find(delta >= dc)';
    clusterInd = find(indNN == ordRho(1));
    indNN_t =indNN(cluster_t);
    clusterOtherInd = find(indNN_t == ordRho(1));
%     clusterOther = indNN_t(clusterOtherInd);
    clusterOther=[];
    for l = 1:Num
        if delta(l)>=dc &&  indNN(l) == ordRho(1)
            clusterOther = [clusterOther;l];
        end
    end
        for l = 1:Num
        if delta(l)>=dc &&  indNN(l) == ordRho(1075)
            clusterOther = [clusterOther;l];
        end
    end

    
    
    [drow,drow_ind] = sort(delta,'descend');
    [rrow,rrow_ind] = sort(rho,'descend');
    
    Num = length(rho);
    N = Num;
    nrho = (rho -min(rho))./ (max(rho)-min(rho));
    nrho = nrho';
    ndelta =(delta-min(delta))./ (max(delta)-min(delta));
    ndelta = ndelta';
    
    %% calculate theta
    rowgamma = nrho.*ndelta;
    gamma_means = mean(rowgamma);
    [gamma, idx]= sort(rowgamma,'descend');
    graph2(N,gamma,ndelta,nrho)
    % x_temp = 1:N;
    % figure()
    % plot(x_temp ,gamma,'r.')
    % figure
    % plot(ndelta ,nrho,'r.');
    
    
    [d,d_ind] = sort(ndelta,'descend');
    [r,r_ind] = sort(nrho,'descend');
    d_gnd = gnd(d_ind);
    d_rho = nrho(d_ind);
    d_gamma = rowgamma(d_ind);
    g_gnd = gnd(idx);
    d=[d,d_ind,d_gnd,d_rho,d_gamma];