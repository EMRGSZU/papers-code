function main_program()

clc;
clear;
close all;
warning('off');
% dataNameArray={'SRBCT','9_Tumors','11_Tumors','Adenocarcinoma','Brain_Tumor1','Brain_Tumor2','Breast3','DLBCL','Leukemia1','Leukemia2','Lung_Cancer','Lymphoma','Nci','Prostate6033','Prostate_Tumor','Brain_Tumor1','Brain_Tumor2','Breast3','RELATHE','PCMAC','BASEHOCK','gisette'};

 dataNameArray={'SRBCT','9_Tumors','11_Tumors','Adenocarcinoma','Brain_Tumor1','Brain_Tumor2','Breast3','DLBCL','Leukemia1','Leukemia2','Lung_Cancer','Lymphoma','Nci','Prostate6033','Prostate_Tumor','Brain_Tumor1','Brain_Tumor2','Breast3','RELATHE','PCMAC','BASEHOCK','gisette'};
for it=1:length(dataNameArray)
    clc
    clearvars -except dataNameArray it
    dataname=dataNameArray{it};
file=['data/',dataname,'.mat'];
load(file);
M=2;
t0=clock;
tloc=5;
CR=0.6;
F=0.5;
fly=1;
for i=1:size(X,2)
    max1=max(X(:,i));
    min1=min(X(:,i));
    X(:,i)=(X(:,i)-min1)/(max1-min1)+0.0001;
end 
X=X;
Y=Y;
X=[Y,X];
[kk,k]=size(X);
Itertime=[];

    
    
    
    
    if k<100
        TT=97;
    else
        TT=290;
    end

    bounds(1:k,1)=0;
    bounds(1:k,2)=1;
    popsize=50;%%������Ⱥ�Ĺ�ģ
    pop=round(rand(popsize,k));%%��ʼ��Ⱥ
    effp=evaluation(pop,k,X);%%ǰ��k����ԭ���ݣ���k+1,k+2����Ŀ�꺯��ֵ

    for t=1:TT
    itertime1=clock;
    C=create(effp,k,F,CR);
    effC=evaluation(C,k,X);%%ǰ��k����ԭ���ݣ���k+1,k+2����Ŀ�꺯��ֵ
    PP=fast_nondominated_sort(effC,effp,M,k);%%��������Ⱥ,���溬��f1f2���У���16��
    
    if ceil(t/tloc)==t/tloc
        AP=sel_pareto(PP,M,k);
        BEST=local_sech3(AP,M,k,X);
        rest=REST(PP,AP,k);
        PP=[rest;BEST];
    end
    
    effp=non_domination_sort_mod1(PP,M,k,popsize);%%���յ���Ⱥ��������Ⱥ�����ĸ�������ͳ�ƣ��ֱ߽磬���PP����Ⱥ����������popsize�������������ӵ���������򣬱������ֵ�
    itertime2=clock;
    Itertime=[Itertime,etime(itertime2,itertime1)];
    t
    disp(etime(itertime2,itertime1));
    
end

    new_AC=sel_pareto(effp,M,k);%%��һ��
    s2=size(new_AC,1);
    AC=[];
    for i=1:s2
        if new_AC(i,k+1)~=0
            AC=[AC;new_AC(i,:)];
        end
    end
    hold on
    plot(AC(:,k+1),AC(:,k+2),'r.');
    xlabel('f1');
    ylabel('f2');
    
    ttt=etime(clock,t0);
    avgacc=1-mean(effp(:,size(effp,2)));
    avgsize=mean(effp(:,size(effp,2)-1));
    avgtime=mean(Itertime);
    savename=[ 'yuan-' dataname '.mat' ];
    save(savename);
end
end