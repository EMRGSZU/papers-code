function fitness_val = fitness(X_all, Y, position)

    if sum(position) == 0
        fitness_val = [1; 1; 1];
        return;
    end

    n_sample = size(X_all, 1);
    n_feature = size(X_all, 2);  % n_feature => full feature
    num_feature = sum(position);  % num_feature => selected feature
    X = X_all(:, position);
    
    % Distance
%     distances = zeros(n_sample);
%     for i = 1:n_sample
%         dis_i = zeros(1, n_sample);
%         for j = i+1:n_sample
% %             dis = sum(abs(X(i, :) - X(j, :))) / num_feature;
%             dis = sqrt(sum((X(i, :) - X(j, :)) .^ 2) / num_feature);
%             dis_i(1, j) = dis;
%         end
%         distances(i, :) = dis_i;
%     end
%     for j = 1:n_sample
%         for i = j+1:n_sample
%             distances(i, j) = distances(j, i);
%         end
%     end
%D=pdist(x) 计算m*n的数据矩阵中对象之间的欧几里得距离。
    D = pdist(X);
    D2 = pdist(X, 'cityblock');
%     disp(X);
%     disp(size(X));
%     disp(D);
%为了节省空间D被格式化为一个向量，但是你可以使用squreform函数吧这个向量转换成一个方阵，
%这样矩阵中的（i,j）i<j,对应于原始数据集中的i和j之间的距离。如下图
    distances = squareform(D) / sqrt(num_feature);
    distances2 = squareform(D2) / num_feature;
%     disp(distances);
    
    % 1NN
    k = 10;
    indices = crossvalind('Kfold', Y, k);
    cp = classperf(Y);
    num_class = size(unique(Y, 'stable'), 1);
    
    for i = 1:k
        test = (indices == i);  % 第i fold的索引？
        train = ~test;  % 索引的补�?
        train_data=X(train,:);
        train_label=Y(train,:);
        test_data=X(test,:);
        
%          class_indices = one_nn(train, test, distances);
         Mdl=fitcknn(train_data,train_label);
%          class = Y(class_indices, :);
        class=predict(Mdl,test_data);
        classperf(cp, class, test);
    end
    error_distribution = cp.ErrorDistributionByClass;
    sample_distribution = cp.SampleDistributionByClass;
    result = sum((sample_distribution - error_distribution) ./ sample_distribution);
    error_rate = 1 - result / num_class;
    
    %1NN
    %mdl=fitcknn(X,Y);
    %cvmdl = crossval(mdl);
    %cvmdlloss = kfoldLoss(cvmdl);
    %error_rate=cvmdlloss;
    
    %SVM
    %Mdl = fitcecoc(X,Y);
    %CVMdl = crossval(Mdl);
    %error_rate = kfoldLoss(CVMdl);
    % 特征比例
    feature_rate = num_feature / size(position, 2);

    % 距离
    db = zeros(1, n_sample);
    dw = zeros(1, n_sample);
    for i = 1:n_sample
        diff_indices = (Y ~= Y(i, :));
        same_indices = find(Y == Y(i, :));
        same_indices = same_indices(same_indices ~= i);
%         disp(sum(diff_indices));
        db(1, i) = min(distances2(i, diff_indices));
        if isempty(same_indices)
            dw(1, i) = 0;
        else
            dw(1, i) = max(distances2(i, same_indices));  % may has no same_indices
        end
    end 
%     disp(db);
    db = sum(db) / n_sample;
    dw = sum(dw) / n_sample;
    distance = 1 / (1 + exp(-5 * (dw - db)));

    fitness_val = [error_rate; feature_rate; distance];

end

function class_indices = one_nn(train, test, distances)
    %返回向量中非零元素的位置 find(A)
    train_indices = find(train);
    test_indices = find(test);
    class_indices = zeros(size(test_indices, 1), 1);
    for k = 1:size(test_indices, 1)  % k为测试样本中的索引�?
        i = test_indices(k);  % i为所有样本中的索�?
        dis_i = distances(i, train);
        [~, min_j] = min(dis_i);
        class_indices(k) = train_indices(min_j);
    end
end
