% function [new_pop, F, params] = select_pop(pop, params)
% 
%     F = nondominated_sort(pop);
% 
%     n_pop = params.n_pop;
%     if numel(pop) == n_pop
%         new_pop = pop;
%         return;
%     end
% 
%     new_pop = [];
%     for l = 1:numel(F)
%         if numel(new_pop) + numel(F{l}) > n_pop
%             Fl = F{l};
%             break;
%         end
%         new_pop = [new_pop; pop(F{l})];
%     end
%     St = [new_pop; pop(Fl)];
%     St_Fl = numel(new_pop)+1:numel(St);
% 
%     [normalized_cost, params] = normalize(St, params);
%     % pi是每个个体关联的参考点索引，d是个体与参考点的距离
%     % 这里需要关联的种群应该是[F1, F2, ..., Fl]，即论文中的St
%     [pi, d] = associate(normalized_cost, params.zr);
%     nzr = size(d, 2);
%     
%     for i = 1:numel(St)
%         St(i).GridIndex = pi(i);
%     end
% 
%     % rho是每个参考点在new_pop中的关联个数
%     rho = zeros(1, nzr);
%     for i = 1:numel(new_pop)
%         rho(pi(i)) = rho(pi(i)) + 1;
%     end
% 
%     while true
%         if numel(new_pop) >= n_pop
%             break;
%         end
%         [~, j] = min(rho);  % St中关联最少的引用点
%         ref_Fl = []; % Fl中与j引用点关联的个体
%         for i = St_Fl
%             if pi(i) == j
%                 ref_Fl = [ref_Fl, i];
%             end
%         end
% 
%         if isempty(ref_Fl)
%             rho(j) = inf;
%             continue;
%         end
% 
%         if rho(j) == 0
%             ddj = d(ref_Fl, j);
%             [~, new_member_index] = min(ddj);  % ref_Fl的索引
%         else
%             new_member_index = randi(numel(ref_Fl));  % ref_Fl的索引
%         end
% 
%         new_member = ref_Fl(new_member_index);  % St中的索引
%         new_pop = [new_pop; St(new_member)];
% 
%         rho(j) = rho(j) + 1;
%         St_Fl(St_Fl == new_member) = [];  % 去掉添加的那个
%     end
% 
%     F = nondominated_sort(new_pop);
% 
% end

function [new_pop, params] = select_pop(pop, params)

    F = nondominated_sort(pop);
    new_pop = pop(F{1});
    
    n_pop = params.n_pop;

    St = new_pop;
    St_Fl = 1:numel(new_pop);

    [normalized_cost, params] = normalize(St, params);
    % pi是每个个体关联的参考点索引，d是个体与参考点的距离
    % 这里需要关联的种群应该是[F1, F2, ..., Fl]，即论文中的St
    [pi, d] = associate(normalized_cost, params.zr);
    for i = 1:numel(new_pop)
        new_pop(i).GridIndex = pi(i);
    end
    
    if numel(new_pop) <= n_pop
        return;
    end
    
    nzr = size(d, 2);

    % rho是每个参考点在new_pop中的关联个数
    rho = zeros(1, nzr);
    for i = 1:numel(new_pop)
        rho(pi(i)) = rho(pi(i)) + 1;
    end

    delete_set = [];
    while true
        if numel(new_pop) - numel(delete_set) == n_pop
            break;
        end
        [~, j] = max(rho);  % St中关联最多的引用点
        ref_Fl = []; % Fl中与j引用点关联的个体
        for i = St_Fl
            if pi(i) == j
                ref_Fl = [ref_Fl, i];
            end
        end

        delete_index = ref_Fl(randi(numel(ref_Fl)));  % ref_Fl的索引
        delete_set = [delete_set delete_index];

        rho(j) = rho(j) - 1;
        St_Fl(St_Fl == delete_index) = [];  % 去掉添加的那个
    end
    new_pop(delete_set) = [];

end

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
            if Dominates(pop(i).Cost, pop(j).Cost)
                domination_set{i} = [domination_set{i}, j];
                dominated_cnt{j} = dominated_cnt{j} + 1;
            end
            if Dominates(pop(j).Cost, pop(i).Cost)
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

function dom=Dominates(x,y)

    if isstruct(x)
        x=x.Cost;
    end

    if isstruct(y)
        y=y.Cost;
    end
    
    dom=all(x<=y) && any(x<y);%计算支配

end