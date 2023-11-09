
M=2;
t0=clock;
tloc=5;
CR=0.6;
F=0.5;
X=data_t;
[kk,k]=size(X);
Itertime=[];
featureacc=zeros(1,k);
featurefre=zeros(1,k);
featurescore=[];
elite=0;
flag=0;
    
    
    
    
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
    effC=evaluation_gai(C,k,X,flag,featurescore);%%ǰ��k����ԭ���ݣ���k+1,k+2����Ŀ�꺯��ֵ
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
    
    for i=1:size(effp,1) 
        featurefre=featurefre+effp(i,1:k);
        featureacc=effp(i,1:k)*effp(i,end)+featureacc;
    end
    elite=elite+size(effp,1);
    if t>=TT/10
        flag=1;
        featurescore=scoresystem2(featurefre,featureacc,elite);
    end
        
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
    disp(mean(Itertime));
    ttt=etime(clock,t0);
save '9_Tumors_gai_test.mat'