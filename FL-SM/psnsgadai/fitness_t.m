function fitness_val = fitness(X_all, Y, position)

    if sum(position) == 0
        fitness_val = [1; 1; 1];
        return;
    end

    n_sample = size(X_all, 1);
    n_feature = size(X_all, 2);  % n_feature => full feature
    num_feature = sum(position);  % num_feature => selected feature
    X = X_all(:, position);
    % 错误率
%     k = size(Y, 1);
    k = 10;
    indices = crossvalind('Kfold', Y, k);
    cp = classperf(Y);
    num_class = size(unique(Y, 'stable'), 1);
    for i = 1:k
        test = (indices == i);  % 第i fold的索引？
        train = ~test;  % 索引的补集
        Mdl = fitcknn(X(train, :), Y(train, :), 'NumNeighbors', 1);
        class = predict(Mdl, X(test, :));
        classperf(cp, class, test);
    end
    error_distribution = cp.ErrorDistributionByClass;
    sample_distribution = cp.SampleDistributionByClass;
    result = sum((sample_distribution - error_distribution) ./ sample_distribution);
    error_rate = 1 - result / num_class;

    % 特征比例
    feature_rate = num_feature / size(position, 2);

    % 距离
    db = [];
    dw = [];
    parfor i = 1:n_sample
        diff_indices = find(Y ~= Y(i, :));
        same_indices = find(Y == Y(i, :));
        same_indices = same_indices(same_indices ~= i);

        d_length = size(diff_indices, 1);
        s_length = size(same_indices, 1);

        db = [db min(sum(abs(repmat(X(i, :), d_length, 1) - X(diff_indices, :)), 2)) / num_feature];
        dw = [dw max(sum(abs(repmat(X(i, :), s_length, 1) - X(same_indices, :)), 2)) / num_feature];
    end
    db = sum(db) / n_sample;
    dw = sum(dw) / n_sample;
    distance = 1 / (1 + exp(-5 * (dw - db)));
    % distance = 1 / (1 + exp(db - dw));
    % disp(['db: ', num2str(db)]);
    % disp(['dw: ', num2str(dw)]);
    % disp(['distance: ', num2str(distance)])

    fitness_val = [error_rate; feature_rate; distance];

end
