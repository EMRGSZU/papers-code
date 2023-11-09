function F = nondominated_sort(pop)
% 非支配排序
    %n=numel（A）；
    %n=numel（A，条件）；
    %返回数组A中元素个数。若是一幅图像，则numel(A)将给出它的像素数。
    num_pop = numel(pop);
    for i = 1:num_pop
        domination_set{i} = [];
        dominated_cnt{i} = 0;
    end

    F{1} = [];
    for i = 1:num_pop
        for j = i+1:num_pop
            if dominates(pop(i).cost, pop(j).cost)
                domination_set{i} = [domination_set{i}, j];
                dominated_cnt{j} = dominated_cnt{j} + 1;
            end
            if dominates(pop(j).cost, pop(i).cost)
                domination_set{j} = [domination_set{j}, i];
                dominated_cnt{i} = dominated_cnt{i} + 1;
            end
        end
        if dominated_cnt{i} == 0
            F{1} = [F{1}, i];
        end
    end

    k = 1;
    while true
        Q = []; % Q用来存储F的k+1层
        for i = F{k}
            for j = domination_set{i}
                dominated_cnt{j} = dominated_cnt{j} - 1;
                if dominated_cnt{j} == 0
                    Q = [Q, j];
                end
            end
        end
        if isempty(Q)
            break;
        end
        F{k + 1} = Q;
        k = k + 1;
    end

end
