function [best_member, F1,Itertime,selectrate,Fittime,Selecttime] = nsga(X,Y,I,T)
 
% load('data/SRBCT.mat');
% X = X;%X读取所有的特征
% Y = Y;%Y读取所有的类别
% X = (X - repmat(min(X), size(X, 1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);


% 特征缩放
% mu = mean(X);  % 每个特征的均值?% sigma = std(X);  % 每个特征的方差?% X_norm = (X - mu) ./ sigma

    % 参数设置
    n_feature = size(X, 2);
    n_obj = 3;
    n_pop = min(round(n_feature / 20), 300);
    n_iter =70;
    n_division = 27;
    n_mutation = n_pop;
    n_crossover = n_pop;
    n_featurelib=25;
    % 产生参考点    
    zr = reference_points(n_obj, n_division);

    % 算法参数
    params.n_pop = n_pop;
    params.zr = zr;
    params.zmin = [];
    params.zmax = [];
    params.smin = [];

    % 初始�?    
    individual.position = [];
    individual.cost = [];
    pop = repmat(individual, n_pop, 1);
    parfor i = 1:n_pop
        position = unifrnd(0, 1, 1, n_feature) > 0.5;
        pop(i).position = position;
        pop(i).cost = fitness(X, Y, position);
%         if mod(i, 100) == 0
%             logger(['init ', num2str(i), '/', num2str(n_pop), ', cost = ', num2str(pop(i).cost')]);
%end
    end

    % 排序
    [pop, F, params] = select_pop(pop, params);
    Fl = pop(F{end});
    avg2 = mean([Fl.cost],2);

    % 迭代
    best_member = [];
    featurerate=Fl(1).cost(2);
    featureerr=Fl(1).cost(1);
    featurelib=zeros(1,n_feature);
    featureacc=zeros(1,n_feature);

    
    subfeaturescore=zeros(1,n_feature);
    featurescore=zeros(1,n_feature);
    
    Itertime=zeros(1,n_iter);
    Fittime=zeros(1,n_iter);
    Selecttime=zeros(1,n_iter);
    selectrate=zeros(1,n_iter);
    
    elite=0;
    cselect=0;
    mselect=0;
    select=0;
    mrate=1;
    avg1=[];
    avg2=[];
    flag_0=0;
    
    pos=[];
%     stop_cnt = 1;
    for iter = 1:n_iter
        iter_time1 = clock;
        
        if iter<n_iter/7
            flag=0;
        %elseif iter>=n_iter*1/7 & iter<n_iter*3/7
        %    flag=1;
        %elseif iter>=n_iter*3/7 & iter<n_iter*6/7
        %    flag=2;
        %else
        %    flag=3;
        end
        
        %if iter>n_iter/7
        %    if select <0.2
        %        flag=1;
        %    elseif select>0.5
        %        flag=3;
        %    else
        %        flag=2;
        %    end
        %end
         
        if iter>n_iter/7 & iter<n_iter*6/7
            if select==0 & flag~=0
                if flag==3
                    flag=2;
                elseif flag==2 
                    flag=1;
                else
                    flag=0;
                end
            else
                if flag==1 & select>0.7
                    flag=3;
                end
            end
            
            if flag==0
                flag_0=flag_0+1;
            end
            
            if flag_0>=3
                flag=3;
                flag_0=0;
            end
        end
        
         if iter>=n_iter*6/7 | mrate<0.01 
             flag=1;
         end
            
        %if flag==1 & select==0
        %    flag=0;
        %end
                
           
        
        if iter==n_iter/7 
            %featurescore=scoresystem1(featurelib,featureacc,n_featurelib,elite,n_feature,featurescore);
            featurescore=scoresystem2(featurelib,featureacc,elite,featurescore);
            avg1=mean([pop.cost], 2);
            featurelib=zeros(1,n_feature);
            featureacc=zeros(1,n_feature);
            elite=0;
            flag=3;
        end
        
        %if iter>n_iter/5 
        %    if featurerate-Fl(1).cost(2)>=0.1 |featureerr-Fl(1).cost(1)>=0.1
        %        subfeaturescore=scoresystem3(featurelib,featureacc,elite,subfeaturescore);
        %        for sub=1:n_feature
        %            if subfeaturescore(sub)<featurescore(sub)
        %                featurescore(sub)=featurescore(sub)*exp((1/sqrt(2))*0.8);
        %            else
        %                featurescore(sub)=featurescore(sub)*exp((1/sqrt(2))*(-0.2));
        %            end
        %        end
        %    end
        %    featureerr=Fl(1).cost(1);
        %    featurerate=Fl(1).cost(2);
        %    featurelib=zeros(1,n_feature);
        %    featureacc=zeros(1,n_feature);
        %    elite=0;
        %end
        
        
        if iter>n_iter/7
            if avg1(1)-avg2(1)>=0.1 || avg1(2)-avg2(2)>=0.05
                %subfeaturescore=scoresystem1(featurelib,featureacc,n_featurelib,elite,n_feature,featurescore);
                subfeaturescore=scoresystem2(featurelib,featureacc,elite,subfeaturescore);
                %featurescore=subfeaturescore;
                avg1(1)=avg2(1);
                avg1(2)=avg2(2);
                %mscore=mean(subfeaturescore);
                pos=[];
                z1=1;
                for sub=1:n_feature
                    if featurescore(sub)~=0
                        if subfeaturescore(sub)<featurescore(sub) &featurescore(sub)~=0
                            featurescore(sub)=0;
                            pos(z1)=sub;
                        else
                            featurescore(sub)=subfeaturescore(sub);
                        end
                    end
                end
                elite=0;
            end
            
        end
        
        % 交叉
        [popc,cselect,fittime1] = normalcrossover(pop, Fl,n_crossover, X, Y,flag,featurescore,pos);
        cselectr=cselect/n_crossover;
        combined_pop = [pop; popc];
        % 变异
        [popm,mselect,fittime2] = normalmutation(pop, Fl,n_mutation, X, Y,flag,featurescore,pos);
        mselectr=mselect/n_mutation;
        combined_pop = [combined_pop; popm];
        select=(cselect+mselect)/(n_crossover+n_mutation);
        
        
        
        
                
        
        if flag~=0
            
            n_cbpop=size(combined_pop,1);
            n_good=0;
            for z=1:n_cbpop
                if combined_pop(z).cost(1)~=1
                    n_good=n_good+1;
                end
            end
            
            
            individual.position = [];
            individual.cost = [];
            good_pop=repmat(individual,n_good, 1);
            
            z1=1;
            
            for k=1:(n_cbpop)
                if combined_pop(k).cost(1)~=1
                    good_pop(z1)=combined_pop(k);
                    z1=z1+1;
                end
            end
            selecttime1=clock;
            [pop, F, params] = select_pop(good_pop, params);
            selecttime2=clock;
            
            F1 = pop(F{1});
            Fl = pop(F{end});
            best_member = analyse(F1);
            
        else
            selecttime1=clock;
            [pop, F, params] = select_pop(combined_pop, params);
            selecttime2=clock;
            
            F1 = pop(F{1});
            Fl = pop(F{end});
            best_member = analyse(F1);
        end
                
        % 选择下一�?        
        
       
        for j=1:size(pop,1)
            featurelib=featurelib+pop(j).position;
            featureacc=featureacc+pop(j).position*(1-pop(j).cost(1));
        end
        elite=elite+j;
        
        
        %for j=1:size(Fl,1)
        %    featurelib=featurelib+Fl(j).position;
        %    featureacc=featureacc+Fl(j).position*(1-Fl(j).cost(1));
        %end
        %elite=elite+j;
       
       

        % 分析
%         best_member = new_best_member;
        iter_time2 = clock;
        Itertime(iter)=etime(iter_time2, iter_time1);
        Fittime(iter)=fittime1+fittime2;
        Selecttime(iter)=etime(selecttime2,selecttime1);
        selectrate(iter)=select;
        
        logger(['Test ',num2str(T),' fold ',num2str(I),'# iteration = ', num2str(iter), '/', num2str(n_iter), ', time = ', num2str(etime(iter_time2, iter_time1)), 's']);
        avgacc = mean([pop.cost], 2);
        avg2 = avgacc;
        mrate=avgacc(2);
        logger(['# avg fitness = ', num2str(avgacc(1)), ', ', num2str(avgacc(2)), ', ', num2str(avgacc(3)),', cselect=',num2str(cselectr),',mselect=',num2str(mselectr)]);
       if iter==9 
           midacc=avgacc(1);
       end
       
       if iter==70
           endacc=avgacc(1);
       end

       
    end
    
 end


function best_member = analyse(F1)
       
    n_F1 = numel(F1);
    best_member = F1(1);
    for i = 2:n_F1
        if F1(i).cost(1) < best_member.cost(1)
            best_member = F1(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) < best_member.cost(2)
            best_member = F1(i);
        end
    end
end
