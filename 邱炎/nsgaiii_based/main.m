function [acc, time]=main(inputpath,outputpath)

rng('shuffle');

Para.gamma=0.2;%杂质参数tempMain
Para.tau=2;%总迭代次数[warning!!!这里应至少改为2]
Para.sigma=1;%相邻样本参数,默认1
Para.n_pop=120;
Para.n_iter=100;
Para.n_obj=3;
Para.n_knn = 1;
%%
data = cell2mat(struct2cell(load(inputpath)));
data(:,2:end) = zscore(data(:,2:end));
indices = crossvalind('Kfold',data(:,1), 10);%10折交叉验证
acc=zeros(1,10);
time=zeros(1,10);
for i=1:10
    test = (indices == i);
    train = ~test;
    Train = data(train, :);
    Test = data(test, :);
    
    TrainLables=Train(:,1);
    TestLables=Test(:,1);
    
    Train=Train(:,2:end);
    Test=Test(:,2:end);
    [N,M] = size(Train);
    fstar=zeros(M,N);
    Ncla = size(unique(TrainLables),1);
    [select_member,best_member,hv_arr, igd_arr] = nsga(Train, TrainLables, Para);
    [SClass]=GH_Classification(Train.',TrainLables.',N, Test.', select_member, Para.gamma, Para.n_knn, Ncla+1);
    [ErClassification]=GH_accuracy(SClass, TestLables.');
    save([outputpath,'-',num2str(i),'.mat'],'select_member','hv_arr', 'igd_arr','ErClassification','TestLables','SClass','indices');
    acc(1,i)=ErClassification;
end
end
