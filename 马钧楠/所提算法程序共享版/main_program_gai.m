
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
    popsize=50;%%定义种群的规模
    pop=round(rand(popsize,k));%%初始种群
    effp=evaluation(pop,k,X);%%前面k列是原数据，第k+1,k+2列是目标函数值

    for t=1:TT
    itertime1=clock;
    C=create(effp,k,F,CR);
    effC=evaluation_gai(C,k,X,flag,featurescore);%%前面k列是原数据，第k+1,k+2列是目标函数值
    PP=fast_nondominated_sort(effC,effp,M,k);%%产生新种群,里面含有f1f2两列，共16列
    
    if ceil(t/tloc)==t/tloc
        AP=sel_pareto(PP,M,k);
        BEST=local_sech3(AP,M,k,X);
        rest=REST(PP,AP,k);
        PP=[rest;BEST];
    end
    
    effp=non_domination_sort_mod1(PP,M,k,popsize);%%最终的种群。对新种群里个体的个数进行统计，分边界，如果PP中种群个数超过了popsize，则对最外层进行拥挤距离排序，保留部分点
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
    
    new_AC=sel_pareto(effp,M,k);%%第一层
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