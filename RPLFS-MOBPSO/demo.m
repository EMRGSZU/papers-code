function [acc]=demo(inputpath,outputpath,Alpha,balanced)

rng('shuffle');

Para.gamma=0.2; % Impurity level
Para.tau=2;% \tau
Para.sigma=1;

data = cell2mat(struct2cell(load(inputpath)));
data(:,2:end) = MINMAX(data(:,2:end)')';
indices = crossvalind('Kfold',data(:,1), 10);%10折交叉验证
acc=zeros(1,10);
for i=1:10
    test = (indices == i);
    train = ~test;
    Train = data(train, :);
    Test = data(test, :);
    
    TrainLables=Train(:,1)';
    TestLables=Test(:,1)';
    
    Train=Train(:,2:end)';
    Test=Test(:,2:end)';

    
    [Para.alpha,~]=size(Train);
    Para.alpha=min([Para.alpha Alpha.max]);
    if Para.alpha<Alpha.min  %alpha下限
        Para.alpha=min(Alpha.min,size(Train,1));
    end

    t1=clock;
    [fstar4,fstarLin4,ErCls4,rClassification4,SClass] = LFS(Train, TrainLables, Test, TestLables, Para,balanced);
    t2=clock;
    t=etime(t2,t1);
    save([outputpath,'-',num2str(i),'.mat'],'fstar4','fstarLin4','ErCls4','rClassification4','t','TestLables','SClass','indices');
    acc(1,i)=rClassification4;
end
end
