

function main()
clc;
clear;
close all;
warning('off');
% parpool(24);
%diary 'parallel_11_Tumors.txt';

% ��ȡ����
%load('data/Prostate6033.mat');
%size(A,n)�����size��������������������һ��n������1��2Ϊn��ֵ��
%�� size�����ؾ��������������������r=size(A,1)����䷵�ص�ʱ����A�������� 
%c=size(A,2) ����䷵�ص�ʱ����A��������
% dataNameArray={'SRBCT','9_Tumors','11_Tumors','Adenocarcinoma','Brain_Tumor1','Brain_Tumor2','Breast3','DLBCL','Leukemia1','Leukemia2','Lung_Cancer','Lymphoma','Nci','Prostate6033','Prostate_Tumor','Brain_Tumor1','Brain_Tumor2','Breast3','RELATHE','PCMAC','BASEHOCK','gisette'};
% ���� 'PCMAC','BASEHOCK',
dataNameArray={'PCMAC','BASEHOCK',};

for it=1:length(dataNameArray)
    clc
    clearvars -except dataNameArray it
dataname=dataNameArray{it};
file=['data/',dataname,'.mat'];
load(file);
X =X;%X��ȡ���е�����
Y =Y ;%Y��ȡ���е����
logger(['num of feature = ', num2str(size(X, 2))]); %����X�����������ж��ٸ�����
total_time1 = clock;

% ��������
% mu = mean(X);  % ÿ�������ľ�ֵ?% sigma = std(X);  % ÿ�������ķ���?% X_norm = (X - mu) ./ sigma
X_norm = (X - repmat(min(X), size(X, 1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);

% ��������
t_max = 1;
result_members_all = [];
Itertimeall=[];
Selectrateall=[];
Fittimeall=[];
Selecttimeall=[];
midacc=[];
endacc=[];
% hvs = []; spacings = [];
for t = 1:t_max

    % k fold cross-validation
    k = 10;
    Itertime=[];
    Selectrate=[];
    Fittime=[];
    Selecttime=[];
    indices = crossvalind('Kfold', Y, k);%indices ��������ı�ǩ
    member.train_acc = 0; member.test_acc = 0; member.n_feature = 0;
    result_members = repmat(member, k, 1);
    for i = 1:k
        logger(['start ', num2str(i), '/', num2str(k), ' fold']);
        fold_time1 = clock;

        test = (indices == i);  % ��i fold������
        train = ~test;  % �����Ĳ�
        % ����ѵ�����Ͳ��Լ�����׼��        
        X_train = X_norm(train, :);
        X_test = X_norm(test, :);
        Y_train = Y(train, :);
        Y_test = Y(test, :);

        % ѡ��
        [best_member, F1, itime,srate,fittime,selecttime]=nsga(X_train, Y_train,i,t);
        
%         [best_member, F1, itime,srate,fittime,selecttime] = nsga(X_train, Y_train);
        X_train = X_train(:, best_member.position);
        X_test = X_test(:, best_member.position);

        % ѵ����Ԥ�⣬kNN������        
        Mdl = fitcknn(X_train, Y_train, 'NumNeighbors', 1);
        class = predict(Mdl, X_test);
        
        % SVM
        %Mdl = fitcecoc(X_train,Y_train);
        %class = predict(Mdl, X_test);

        % ���ȷ���
        cp = classperf(Y_test);
        num_class = size(unique(Y_test, 'stable'), 1);
        classperf(cp, class);
        error_distribution = cp.ErrorDistributionByClass;
        sample_distribution = cp.SampleDistributionByClass;
        result = sum((sample_distribution - error_distribution) ./ sample_distribution);
        accuracy = result / num_class;

        % ����ѡ��
        result_members(i).train_acc = 1 - best_member.cost(1);
        result_members(i).test_acc = accuracy;
        result_members(i).n_feature = sum(best_member.position);

        % ����
        fold_time2 = clock;
        Itertime=[Itertime;itime];
        Selectrate=[Selectrate;srate];
        Fittime=[Fittime;fittime];
        Selecttime=[Selecttime;selecttime];
        PopObj = [F1.cost]';
       % hv = HV(PopObj, [0, 0, 0]);
       % spacing = Spacing(PopObj);
       % hvs = [hvs; hv]; spacings = [spacings; spacing];
        logger(['## fold ', num2str(i), '/', num2str(k), ', time = ', num2str(etime(fold_time2, fold_time1)), 's']);
        logger(['## max training accuracy = ', num2str(result_members(i).train_acc)]);
        logger(['## max testing accuracy = ', num2str(result_members(i).test_acc)]);
        logger(['## feature size = ', num2str(result_members(i).n_feature)]);
        %logger(['## spacing = ', num2str(spacing), ', hv = ', num2str(hv)]);
    end
    % ����ÿ�����еĽ��    
    result_members_all = [result_members_all; result_members];
    Itertimeall=[Itertimeall;sum(Itertime)./k];
    Selectrateall=[Selectrateall;sum((Selectrate)./k)];
    Fittimeall=[Fittimeall;sum((Fittime)./k)];
    Selecttimeall=[Selecttimeall;sum((Selecttime)./k)];
    logger(['### round', num2str(t), ' complete']);
end

% ����
total_time2 = clock;
total_time = etime(total_time2, total_time1);
if t_max==1
    mtimeall=Itertimeall/t_max;
    mrateall=Selectrateall/t_max;
    mfittimeall=Fittimeall/t_max;
    mselecttimeall=Selecttimeall/t_max;
else
    mtimeall=sum(Itertimeall,1)/t_max;
    mrateall=sum(Selectrateall,1)/t_max;
    mfittimeall=sum(Fittimeall,1)/t_max;
    mselecttimeall=sum(Selecttimeall,1)/t_max;
end
m_itertime=sum(mtimeall)/70;
m_fittime=sum(mfittimeall)/70;
m_selecttime=sum(mselecttimeall)/70;


logger(['total time consumed = ', num2str(total_time), 's / ', num2str(total_time/60), 'min']);
mean_train_acc = mean([result_members_all.train_acc]);
mean_test_acc = mean([result_members_all.test_acc]);
mean_feature_size = mean([result_members_all.n_feature]);
std_test_acc = std([result_members_all.test_acc] * 100);
std_size=std([result_members_all.n_feature]);
xx=reshape(Itertime,[],1);
std_time=std(xx);
dlmwrite('result_members_all.txt', cell2mat(struct2cell(result_members_all))')
logger(['average iter time = ',num2str(m_itertime)]);
logger(['average training accuracy = ', num2str(mean_train_acc)]);
logger(['average testing accuracy = ', num2str(mean_test_acc)]);
logger(['std testing accuracy = ', num2str(std_test_acc)]);
logger(['std size  = ', num2str(std_size)]);
logger(['std time = ', num2str(std_time)]);
logger(['average feature size = ', num2str(mean_feature_size)]);
savename=[ 'test2' dataname '.mat' ];
save(savename);

end
end
