%function [best_member, F1,Itertime] = nsga(X, Y)
    
load('data/SRbct.mat');
X = data(:, 2:end);%X读取所有的特征
Y = data(:, 1);%Y读取所有的类别
logger(['num of feature = ', num2str(size(X, 2))]); %返回X的列数，即有多少个特征
total_time1 = clock;

% 特征缩放
% mu = mean(X);  % 每个特征的均值?% sigma = std(X);  % 每个特征的方差?% X_norm = (X - mu) ./ sigma
X_norm = (X - repmat(min(X), size(X, 1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);
X=X_norm;
    % 参数设置
    n_feature = size(X, 2);
    n_obj = 3;
    n_pop = min(round(n_feature / 20), 300);
    n_iter = 20;
    % n_pop = 3; n_iter = 2;
    n_division = 27;
%     n_mutation = ceil(n_pop / 2);
%     n_crossover = floor(n_pop / 2);
    n_mutation = n_pop;
    n_crossover = n_pop;
    n_featurelib=25;
    % 产生参考点    
    zr = reference_points(n_obj, n_division);

    % 算法参数
    params.n_pop = n_pop;
    params.zr = zr;
    params.zmin = [];
    params.zmax = [];
    params.smin = [];

    % 初始�?    
    individual.position = [];
    individual.cost = [];
    pop = repmat(individual, n_pop, 1);
    parfor i = 1:n_pop
        position = unifrnd(0, 1, 1, n_feature) > 0.5;
        pop(i).position = position;
        pop(i).cost = fitness(X, Y, position);
%         if mod(i, 100) == 0
%             logger(['init ', num2str(i), '/', num2str(n_pop), ', cost = ', num2str(pop(i).cost')]);
%end
    end

    % 排序
    [pop, F, params] = select_pop(pop, params);
    Fl = pop(F{end});

    % 迭代
    best_member = [];
    featurelib=zeros(1,n_feature);
    featureacc=zeros(1,n_feature);
    featurescore=zeros(1,n_feature);
    Itertime=zeros(1,n_iter);
    elite=0;
%     stop_cnt = 1;
    for iter = 1:n_iter
        iter_time1 = clock;
        flag=0;
        if iter<n_iter/5
            flag=1;
        
        end
        if iter==n_iter/5
            nF1=size(F1,1);
            nclass=size(unique(Y, 'stable'), 1);
            H=zeros(1,n_feature);
            HFC=zeros(nclass,n_feature);
            HC=zeros(nclass,1);
            P1=zeros(1,n_feature);
            SU=zeros(nclass,n_feature);
            for m=1:nF1
                P1=P1+F1(m).position;
                HFC(Y(m)+1,:)=HFC(Y(m)+1,:)+F1(m).position;
                HC(Y(m)+1)=HC(Y(m)+1)+1;
            end
            for p=1:nclass
                for q=1:n_feature
                    p1=HFC(p,q)/HC(p);
                    p0=1-p1;
                    HC(p)=HC(p)/sum(HC);
                    HFC(p,q)=(-HC(p))*(p1*log2(p1)+p0*log2(p0));
                    if p==1
                        p1=P1(q)/nF1;
                        p0=1-p1;
                        H(q)=-(p1*log2(p1)+p0*log2(p0));
                    end
                    SU(p,q)=(H(q)-HFC(p,q))/(H(q)+HC(p));
                end
            end
            for p=1:nclass
                for q=p:nclass
                    featurescore=featurescore+abs(SU(p)-SU(q));
                end
            end
            featurescore=featurescore*(featureacc/featurelib);
        end
        % 交叉
        popc = crossover_pop(pop, n_crossover, X, Y,flag,featurescore);
        combined_pop = [pop; popc];
        % 变异
        popm = mutate_pop(combined_pop, n_mutation, X, Y, Fl,flag,featurescore);
        combined_pop = [combined_pop; popm];
        % 选择下一�?        
        [pop, F, params] = select_pop(combined_pop, params);
        F1 = pop(F{1});
        Fl = pop(F{end});
        best_member = analyse(F1);
        if iter<n_iter/5
            for j=1:size(F1,1)
                featurelib=featurelib+F1(j).position;
                featureacc=featureacc+F1(j).position*(1-F1(j).cost(1));
            end
            elite=elite+j;
        end

        % best_member如果10次未变，就停止以避免过拟�?        
        % if ~isempty(best_member)
        %     if all(new_best_member.cost == best_member.cost)
        %         stop_cnt = stop_cnt + 1;
        %     else
        %         stop_cnt = 1;
        %     end
        % end

        % 分析
%         best_member = new_best_member;
        iter_time2 = clock;
        Itertime(iter)=etime(iter_time2, iter_time1);
        logger(['# iteration = ', num2str(iter), '/', num2str(n_iter), ', time = ', num2str(etime(iter_time2, iter_time1)), 's']);
        avgacc = mean([pop.cost], 2);
        logger(['# avg fitness = ', num2str(avgacc(1)), ', ', num2str(avgacc(2)), ', ', num2str(avgacc(3))]);
        % logger(['## accuracy = ', num2str(1 - best_member.cost(1)), ', features = ', num2str(round(n_feature * best_member.cost(2)))]);

        % if stop_cnt >= 10
        %     logger('stopping criterion reached');
        %     break;
        % end
    end
%end

function best_member = analyse(F1)
    % 挑�?精度�?��的个体中距离�?���?    
    n_F1 = numel(F1);
    best_member = F1(1);
    for i = 2:n_F1
        if F1(i).cost(1) < best_member.cost(1)
            best_member = F1(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) < best_member.cost(2)
            best_member = F1(i);
        end
    end
end
