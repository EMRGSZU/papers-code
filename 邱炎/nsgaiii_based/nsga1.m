function [best_member, F1,Itertime,Fittime,Selecttime] = nsga1(X, Y)
    % 参数设置
    [n_number, n_feature] = size(X);
    n_obj = 3;
    %n_pop = min(round(n_feature / 20), 300);
    n_pop = 150;
    n_iter = 70;
    % n_pop = 3; n_iter = 2;
    n_division = 27;
%     n_mutation = ceil(n_pop / 2);
%     n_crossover = floor(n_pop / 2);
    n_mutation = n_pop;
    n_crossover = n_pop;

    % 产生参考点    
    zr = reference_points(n_obj, n_division);

    % 算法参数
    param.n_pop = n_pop;
    param.zr = zr;
    param.zmin = [];
    param.zmax = [];
    param.smin = []; 
    params = repmat(param,n_number,1);
    
    % 初始�?    
    individual.position = [];
    individual.cost = [];
    individual.crossPoint = [];
    pop = repmat(individual, n_number, n_pop);
    for i = 1:n_number
        parfor j = 1:n_pop
            position = unifrnd(0, 1, 1, n_feature) > 0.5;
            pop(i,j).position = position;
            [pop(i,j).cost, pop(i,j).crossPoint] = fitness(i, X, Y, position);
        end
       %排序
       [pop_temp, ~ , params(i)] = select_pop(pop(i,:).', params(i));
       pop(i,:) = pop_temp.';
    end


    % 迭代   
    
    best_member = repmat(individual, n_iter, n_number);
    
    Itertime=zeros(n_iter,n_number);
    Fittime=zeros(n_iter,n_number);
    Selecttime=zeros(n_iter,n_number);
    selectrate=zeros(n_iter,n_number);
    
    
%     stop_cnt = 1;
    for iter = 1:n_iter
        iter_time1 = clock;
        for num = 1:n_number
            % 交叉
            [popc,~] = normalcrossover(pop(num,:).', num, n_crossover, X, Y);
            combined_pop = [pop(num,:).'; popc];
            % 变异
            [popm,~] = normalmutation(pop(num,:).', num, n_mutation, X, Y);
            combined_pop = [combined_pop; popm];
            % 选择下一�?        
            %selecttime1=clock;
            [pop_temp, F, params(num)] = select_pop(combined_pop, params(num));
            %selecttime2=clock;
            pop(num,:) = pop_temp.';
            F1 = pop_temp(F{1});
            [best_member(iter,num),~] = analyse(F1 , F{1}); %因为要替换掉原种群中best的那个位置，所以加入best_pop代表最好的种群
            %if num == n_number
            %    best_member(iter,:) = get_Push(best_member(iter,:),n_number,X, Y);
            %    pop(:,end) = best_member(iter,:);
            %end
            %if num == n_number
            %    best_member(iter,:) = get_Pop(best_member(iter,:),n_number,X, Y);
            %    pop(:,end) = best_member(iter,:);
            %end
      
            %Itertime(iter,num)=etime(iter_time2, iter_time1); %迭代所花的时间 
            %Fittime(iter,num)=fittime1+fittime2;  %变异所花费的时间
            %Selecttime(iter,num)=etime(selecttime2,selecttime1); %挑选种群所花费时间
            avgacc = mean([pop(num,:).cost], 2);
            logger(['# avg fitness = ', num2str(avgacc(1)), ', ', num2str(avgacc(2)), ', ', num2str(avgacc(3))]);
            logger([ 'iteration - ',num2str(iter), '/loactin - ', num2str(num), '# best_fitness = ', num2str(best_member(iter,num).cost(1)), ', ', num2str(best_member(iter,num).cost(2)), ', ', num2str(best_member(iter,num).cost(3))]);
        end
        iter_time2 = clock;
        %Itertime(iter)=etime(iter_time2, iter_time1); %迭代所花的时间 
        %Fittime(iter)=fittime1+fittime2;  %变异所花费的时间
        %Selecttime(iter)=etime(selecttime2,selecttime1); %挑选种群所花费时间
        logger(['# iteration = ', num2str(iter), '/', num2str(n_iter), ', time = ', num2str(etime(iter_time2, iter_time1)), 's']);
    end
end

function [best_member,best_pop] = analyse(F1, not_dominate)
    n_F1 = numel(F1);
    best_member = F1(1);
    best_pop = not_dominate(1);
    for i = 2:n_F1
        if F1(i).cost(1) < best_member.cost(1)
            best_member = F1(i);
            best_pop = not_dominate(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) < best_member.cost(2)
            best_member = F1(i);
            best_pop = not_dominate(i);
        end
    end
end
