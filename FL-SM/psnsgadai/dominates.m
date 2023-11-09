function d = dominates(cost_1, cost_2)
    if cost_1(1) + 0.2 <= cost_2(1)
        d = true;
%     elseif cost_1(1) == cost_2(1)
%         d = false;
    else
        %all函数：检测矩阵中是否全为非零元素，如果是，则返回1，否则，返回0。
        %any函数：检测矩阵中是否有非零元素，如果有，则返回1，否则，返回0。用法和all一样
        %A&&B 首先判断A的逻辑值，如果A的值为假，就可以判断整个表达式的值为假，
        %就不需要再判断B的值。这种用法非常有用，如果A是一个计算量较小的函数，B是一个计算量较大的函数，
        %那么首先判断A对减少计算量是有好处的。另外这也可以防止类似被0除的错误：
        d = all(cost_1 <= cost_2) && any(cost_1 < cost_2);
    end
end
