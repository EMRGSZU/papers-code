function [new_best_member] = get_Pop(best_member, N, X, Y)
new_best_member = best_member;
    for i = 1:N
       new_postion = best_member(i).position;
       position_num = sum(new_postion);
       crossPosition = best_member(i).position;   %取源数据的position位置
       crossPoint = best_member(i).crossPoint;    %取出源数据中position最优的位置
       crossNum = size(crossPoint,2);          %
       for j = 1:crossNum
           crossPosition = crossPosition + best_member(crossPoint(j)).position;
       end
       origin_position = find(best_member(i).position == 1);
       count_origin_position = crossPosition(origin_position);  %统计源数组中特征出现的个数
       num_of_test = sum(count_origin_position~=0);
       [~, index] = sort(count_origin_position);
       if num_of_test > 20
           num_of_test = 20;
       end
       for j = 1: num_of_test
           new_postion(origin_position(index(j))) = 0;
           [fitness_val,cross_point] = fitness(i, X, Y, new_postion);
           %if dominates(fitness_val,best_member(i).cost)
           if fitness_val(1) < best_member(i).cost(1)||( fitness_val(1) == best_member(i).cost(1) && fitness_val(2) < best_member(i).cost(2))
              new_best_member(i).position = new_postion;
              new_best_member(i).cost = fitness_val;
              new_best_member(i).crossPoint = cross_point;
              break
           else
              new_postion(origin_position(index(j))) = 1;
           end   
       end
    end
end

