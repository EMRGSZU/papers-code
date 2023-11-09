function fitness_val = fitness(X_all, Y, position)
    position=logical(position);
    X = X_all(:, position);
      
    % 1NN
    k = 10;
    indices = crossvalind('Kfold', Y, k);
    cp = classperf(Y);
    num_class = size(unique(Y, 'stable'), 1);
    
    for i = 1:k
        test = (indices == i);  
        train = ~test;  
        train_data=X(train,:);
        train_label=Y(train,:);
        test_data=X(test,:);
        
        Mdl=fitcknn(train_data,train_label);
        class=predict(Mdl,test_data);
        classperf(cp, class, test);
    end
    error_distribution = cp.ErrorDistributionByClass;
    sample_distribution = cp.SampleDistributionByClass;
    result = sum((sample_distribution - error_distribution) ./ sample_distribution);
    error_rate = 1 - result / num_class;
    

   
    fitness_val = error_rate;

end

