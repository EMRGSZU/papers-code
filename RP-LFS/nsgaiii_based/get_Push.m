function [new_best_member] = get_Push(best_member, N, X, Y)
new_best_member = best_member;
position_num = size(X,2);
    for i = 1:N
       new_postion = best_member(i).position;
       crossPosition = best_member(i).position;   %取源数据的position位置
       crossPoint = best_member(i).crossPoint;    %取出源数据中position最优的位置
       crossNum = size(crossPoint,2);          %
       for j = 1:crossNum
           crossPosition = crossPosition + best_member(crossPoint(j)).position;
       end
       origin_position = find(best_member(i).position == 1);
       crossPosition(origin_position) = 0;
       num_of_test = sum(crossPosition~=0);
       [~, index] = sort(crossPosition,'descend');  
        if num_of_test > 20
           num_of_test = 20;
       end
       for j = 1: num_of_test
           if crossPosition(index(j)) == 0
              break 
           end
           new_postion(index(j)) = 1;
           [fitness_val,cross_point] = fitness(i, X, Y, new_postion);
           %if dominates(fitness_val,best_member(i).cost)
           if fitness_val(1) < best_member(i).cost(1) || ( fitness_val(1) == best_member(i).cost(1) && fitness_val(2) < best_member(i).cost(2))
              new_best_member(i).position = new_postion;
              new_best_member(i).cost = fitness_val;
              new_best_member(i).crossPoint = cross_point;
              break
           else
              new_postion(index(j)) = 0;  
           end   
       end
    end
end

