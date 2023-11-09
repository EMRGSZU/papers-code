function [numClust,clustInd,centInd] = ADPC(rho,dist)

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
    
%     isManualSelect = 1; % 1 denote that all the cluster centroids are selected manually, otherwise 0
    [numClust, centInd] = ADPC_Graph(rho, delta,dist,ordRho);
    
    % after the cluster centers have been found,
    % each remaining point is assigned to the same cluster as its nearest neighbor of higher density
    clustInd = zeros(1, Num);
    for i = 1 : Num
        if centInd(ordRho(i)) == 0 % not centroid
            clustInd(ordRho(i)) = clustInd(indNearNeigh(ordRho(i)));
        else
            clustInd(ordRho(i)) = centInd(ordRho(i));
        end
    end
    
end

function [numClust, centInd] = ADPC_Graph(rho, delta,dist,ordrho)
%%DECISIONGRAPH Decision graph for choosing the cluster centroids.
%   INPUT:
%       rho: local density [row vector]
%       delta: minimum distance between each point and any other point with higher density [row vector]
%       isManualSelect: 1 denote that all the cluster centroids are selected manually, otherwise 0
%  OUTPUT:
%       numClust: number of clusters
%       centInd:  centroid index vector
    Num = length(rho);
    numClust = 0;
    centInd = zeros(1, Num);
    N = Num;
    nrho = (rho -min(rho))./ (max(rho)-min(rho));
    nrho = nrho';
    ndelta =(delta-min(delta))./ (max(delta)-min(delta));
    ndelta = ndelta';
%     nrho(find(nrho == 0 )) = 0.0000001;
%     ndelta(find(ndelta == 0))= 0.0000001;
%% calculate theta
    gamma =min(nrho,ndelta);
    gamma_means = mean(gamma);
    [gamma idx]= sort(gamma,'descend');

%% 通过决策图密度确定聚类中心以及聚类数
slope=[];
for i=1:N-1
    slope = [slope; gamma(i)-gamma(i+1)];
end
slope_gap = [];
for i = 1:N-2
    slope_gap = [slope_gap; abs( abs(slope(i)) - abs(slope(i+1)) )];
end
beta = sum(slope_gap)/(N-2);    %平均斜率差
slope_rate = [];
for i = 1:N-2
    slope_rate = [slope_rate; slope(i)/slope(i+1)];
end
slope_rate_mean = sum(slope_rate)/(N-2);

three_points_slope = [];
for i = 1:N-2
    three_points_slope = [three_points_slope; slope(i)+slope(i+1)];
end

p = idx(find(slope_gap >= beta));
suspected_points = find(slope_gap >= beta);
suspected_points = find(suspected_points <= 100);


k=10
gamma_extend = repmat(gamma,1,N);
gamma_dist = abs(gamma_extend - gamma_extend');
gamma_dist_sorted = sort(gamma_dist,2);
gamma_delta_k = gamma_dist_sorted(:,k+1);    % a volumn vector
gamma_mu_k = sum(gamma_delta_k)/N;
gamma_dc = gamma_mu_k + sqrt((1/(N-1))*sum(((gamma_delta_k - gamma_mu_k).^2)));
gamma_KNN = gamma_dist_sorted(:,2:k+1);
gamma_rho = sum(exp(-(gamma_KNN.^2)/(gamma_dc^2)),2);
gamma_rho_mean = sum(gamma_rho)/size(gamma_rho,1);


gamma_rho_gap = [];
for i = 1: size(gamma_rho,1)-1
    gamma_rho_gap = [gamma_rho_gap; abs(gamma_rho(i+1)-gamma_rho(i))];
end
gamma_rho_gap = gamma_rho_gap(suspected_points);

% 得到疑似聚类数点后，判断该点之后的数据是否趋于稳定
%假定计算稳定度用的k为大于gamma均值的2倍以上的点的数量
% k = size(find(gamma > 2*gamma_means),1)
k=20

stable_rate = [];
for i = 1: N-k
    stable_rate = [stable_rate; 1/(gamma(i+1)-gamma(i+k))];
end
suspected_stable_rate = stable_rate(suspected_points,:);
% [~,ssr_idx] = sort(suspected_stable_rate);
% ssr_score=0:(1./size(suspected_stable_rate)):1;
% ssr_score = ssr_score';
% suspected_stable_rate = ssr_score(ssr_idx,:)

gamma_rho_gap = mapminmax(gamma_rho_gap',0,1)';
gamma_rho_gap = sqrt(2)*gamma_rho_gap
suspected_stable_rate = mapminmax(suspected_stable_rate',0,1)'
% suspected_slope_gap = slope_gap(suspected_points-1,:);
% suspected_slope_gap = mapminmax(suspected_slope_gap',0,1)';
% suspected_slope_rate = slope_rate(suspected_points-1,:);
% suspected_slope_rate = mapminmax(suspected_slope_rate',0,1)';

% 利用密度计算权重值
weight_rate = gamma_rho_gap+suspected_stable_rate

%利用斜率差计算权重值
% weight_rate = suspected_slope_gap + stable_rate

%利用斜率差+斜率比例+稳定度计算权重值
% weight_rate = suspected_slope_gap+suspected_slope_rate+suspected_stable_rate+suspected_stable_rate_gap

% tend = [];
% for i = 2:N-1
%     tend = [tend;(i-1)*(gamma(i+1)-gamma(i))/(gamma(i)-gamma(1))];
% end
% suspected_tend = tend(suspected_points+1,:);
% suspected_tend = mapminmax(suspected_tend',0,1)'



[~,clusters] = max(weight_rate)
center_idxs = idx(find(gamma >= gamma(clusters)))
% clusters = 3;
% center_idxs = idx(find(gamma >= gamma(clusters)))


% 绘制gamma决策图
fig = figure(1);
% set(fig,'units','normalized','position',[0 0 0.6 0.9]);
plot(gamma(1:100,:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k')
hold
% plot(clusters,gamma(clusters),'o','MarkerSize',5,'MarkerFaceColor','r','MarkerEdgeColor','r')
% plot(suspected_points,gamma(suspected_points,:),'o','MarkerSize',5,'MarkerFaceColor','r','MarkerEdgeColor','r')
title ('Decision Graph(\gamma)','FontSize',15.0)
xlabel ('n')
ylabel ('\gamma')

figure(200);
plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
xlabel('\rho');
ylabel('\delta');
title('Decision Graph','FontSize',15.0)
numClust = clusters;
%% 进行聚类
% raw assignment
disp('Assigning data-points into clusters...');
cluster_lables = -1*ones(size(dist,1),1);
for i = 1:length(center_idxs)
    cluster_lables(center_idxs(i)) = i;
end
for i=1:length(cluster_lables)
    if (cluster_lables(ordrho(i))==-1)
        cluster_lables(ordrho(i)) = cluster_lables(nneigh(ordrho(i)));
    end
end
%     for i = 1 : NE
%         if (theta(i) > x_hat)
%             numClust = numClust + 1;
%             centInd(i) = numClust;
%         end
%     end
% 
% %% draw the judge diagram
% %         fprintf('Manually select a proper rectangle to determine all the cluster centres (use Decision Graph)!\n');
% %         fprintf('The only points of relatively high *rho* and high  *delta* are the cluster centers!\n');
% 
%         [theta_sorted,~] = sort(theta);
%         h1 = plot(theta_sorted, 'o', 'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5);
%         xlabel('n','FontSize',20,'Fontname','Times newman');
%         ylabel('\theta','Rotation',0,'FontSize',20,'Fontname','Times newman');
%         title('Judge Diagram','FontSize',15.0)
% 
%         hold on
%         h2 = plot(xlim,[x_hat,x_hat],'r','linewidth',2);
%         legend(h2,'DPC GP')
%     
%         
%         
%         % DO NOTHING, just for futher work ...
end