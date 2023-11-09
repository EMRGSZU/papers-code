clear;
load('data/wine.mat');
X = DataSet(:, 2:end);%X读取所有的特征
Y = DataSet(:, 1);%Y读取所有的类别


Para.gamma=0.2;%杂质参数tempMain
Para.tau=2;%总迭代次数[warning!!!这里应至少改为2]
Para.sigma=1;%相邻样本参数,默认1
Para.n_pop=10;
Para.n_iter=3;
Para.n_obj=3;
Para.n_knn = 1;
%修改了dominate的约束0.2 and

%使用了加入的crossover

%mutation 也用回了原来那个
%only test for crossover.
t_max = 1;
gamma = 0.2;
iteration = 3;
knn =1;
result_members_all1 =  [];
%result_members_all2 = [];
diff_iter_feature_num1 = [];
diff_iter_dp = [];
diff_iter_distance = [];
%diff_iter_feature_num2 = [];
diff_iter_results1 = [];
%diff_iter_results2 = [];
X_norm = (X - repmat(min(X), size(X, 1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);
Ncla = 5;
t1=clock;
for t = 1:t_max
    % k fold cross-validation
    k = 10;
    indices = crossvalind('Kfold', Y, k);%indices 就是这个的标签
    member.acc = 0; member.balance_acc = 0; member.feature = 0;
    result_members1 = repmat(member, k, 1);
    result_members2 = repmat(member, k, 1);
    parfor i = 1:k
        total_feature = 0;
        test = (indices == i);  % 第i fold的索引
        train = ~test;  % 索引的补
        % 划分训练集和测试集，标准化        
        X_train = X_norm(train, :);
        X_test = X_norm(test, :);
        Y_train = Y(train, :);
        Y_test = Y(test, :);
        cp = classperf(Y_test);
        num_class = size(unique(Y_test, 'stable'), 1);
        [best_member1,normal_best_member1 ,hv_arr, igd_arr] = nsga(X_train, Y_train, Para);
        N = size(X_train,1);
        [S_Class1]=GH_Classification(X_train.',Y_train.',N, X_test.',best_member1(iteration,:),gamma, knn, Ncla);
        [ErClassification1]=GH_accuracy(S_Class1, Y_test.');
        classperf(cp, S_Class1.');
        error_distribution = cp.ErrorDistributionByClass;
        sample_distribution = cp.SampleDistributionByClass;
        result = sum((sample_distribution - error_distribution) ./ sample_distribution);
        accuracy = result / num_class;
        result_members1(i).acc = ErClassification1;
        result_members1(i).balance_acc = accuracy;
        result_members1(i).feature = total_feature / N;
        logger([ 'iteration - ',num2str(i),'/acc - ErClassification - ',num2str(ErClassification1)]);
        %%partII
        %[best_member2,~,~,~,~] = nsga1(X_train, Y_train);
        %N = size(X_train,1);
        %[S_Class2, feature_num2]=GH_Classification(X_train.',Y_train.',N, X_test.',best_member2(iteration,:),gamma, knn, Ncla);
        %[ErClassification2]=GH_accuracy(S_Class2, Y_test.');
        %for iter = 1:1:iteration
        %    [iter_S_Class2,iter_feature_num2]=GH_Classification(X_train.',Y_train.',N, X_test.',best_member2(iter,:),gamma, knn, Ncla);
        %    [iter_ErClassification2]=GH_accuracy(iter_S_Class2, Y_test.');
        %    diff_iter_results2 = [diff_iter_results2,iter_ErClassification2];
        %    diff_iter_feature_num2 = [diff_iter_feature_num2,iter_feature_num2];
        %end
        %classperf(cp, S_Class2.');
        %error_distribution = cp.ErrorDistributionByClass;
        %sample_distribution = cp.SampleDistributionByClass;
        %result = sum((sample_distribution - error_distribution) ./ sample_distribution);
        %accuracy = result / num_class;
        %result_members2(i).acc = ErClassification2;
        %result_members2(i).balance_acc = accuracy;
        %result_members2(i).feature = feature_num2;
        %logger([ 'iteration - ',num2str(i),'/acc - ErClassification - ',num2str(ErClassification2)]);            
    end
    %diff_iter_results = [diff_iter_results,iter_members];
    result_members_all1 = [result_members_all1; result_members1];
    %result_members_all2 = [result_members_all2; result_members2];
end
t2 = clock;
logger([ 'total- time ',num2str(etime(t2,t1))]);   
dlmwrite('result_members_all.txt', cell2mat(struct2cell(result_members_all1))')
%dlmwrite('diff_iter_results.txt',diff_iter_results1);
mean_acc = mean([result_members_all1.balance_acc]);
mean_std = std([result_members_all1.balance_acc]);